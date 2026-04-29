package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// AnthropicModel represents an Anthropic model.
type AnthropicModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *AnthropicModel) ID() string {
	return m.id
}

// Provider returns the provider identifier. Defaults to "anthropic"; derived
// providers (Bedrock Anthropic, Vertex Anthropic) override it via Config.
func (m *AnthropicModel) Provider() string {
	return m.provider.cfg.providerID()
}

// Stream sends a streaming request to Anthropic.
func (m *AnthropicModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *AnthropicModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Determine whether the synthetic "json" tool will be injected so the
	// stream processor can surface its input as text. Decided here (not
	// inside buildRequest) so the stream layer and the request layer stay
	// in sync without shared state.
	jsonToolInjected := false
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" && len(options.ResponseFormat.Schema) > 0 {
		jsonToolInjected = true
	}

	reqBody, betas, warnings, err := BuildRequestBody(m.provider.cfg, m.id, options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := m.provider.opts.BaseURL + "/messages"
	if m.provider.cfg.BuildRequestURL != nil {
		url = m.provider.cfg.BuildRequestURL(m.provider.opts.BaseURL, true)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	switch m.provider.opts.AuthScheme {
	case "bearer":
		req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	default:
		req.Header.Set("x-api-key", m.provider.opts.APIKey)
	}
	req.Header.Set("anthropic-version", apiVersion)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	// Set anthropic-beta header when the configured provider expects betas
	// out-of-band (direct Anthropic, Vertex). Bedrock emits them inside the
	// body via TransformRequestBody + EmitBetasInBody.
	if !m.provider.cfg.EmitBetasInBody && len(betas) > 0 {
		req.Header.Set("anthropic-beta", strings.Join(betas, ","))
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{
			Error: fmt.Errorf("Anthropic API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, jsonToolInjected, options.IncludeRawChunks)
}

// syntheticJSONToolName is the name of the synthetic tool injected when the
// caller requests structured JSON output. Its input IS the structured result;
// the stream processor surfaces it as text so generic Output parsing works.
const syntheticJSONToolName = "json"

// buildRequest is a thin wrapper around BuildRequestBody kept for existing
// tests that call m.buildRequest directly.
func (m *AnthropicModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	body, _, warnings, err := BuildRequestBody(m.provider.cfg, m.id, options)
	return body, warnings, err
}

// BuildRequestBody assembles an Anthropic Messages request body for the given
// model + call options. It returns the marshalled JSON, the collected
// anthropic-beta tokens (deduplicated, order-preserving), warnings, and any
// error. Consumers (direct Anthropic, Bedrock, Vertex) can call this from
// their own Stream path and then dispatch the body on their own transport.
func BuildRequestBody(cfg Config, modelID string, options *stream.CallOptions) ([]byte, []string, []stream.Warning, error) {
	var warnings []stream.Warning

	opts, err := provider.ParseProviderOptions[MessagesOptions](options.ProviderOptions)
	if err != nil {
		return nil, nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on Anthropic (ai-sdk parity):
	// frequencyPenalty, presencePenalty, seed.
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if options.PresencePenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", ""))
	}
	if options.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}

	inputMessages := options.Messages

	// Resolve structured-output capability. When the provider can't take a
	// native output_config.format payload, force synthetic JSON tool
	// injection even if the caller requested outputFormat mode.
	useNativeStructuredOutput := cfg.supportsNativeStructuredOutput()

	// ResponseFormat mapping.
	// - Schema: inject synthetic "json" tool + force tool_choice (when
	//   native structured output is disabled OR caller didn't opt in via
	//   structuredOutputMode=outputFormat).
	// - No schema: emit a warning and inject a system-prompt JSON
	//   instruction.
	useOutputFormat := useNativeStructuredOutput && opts.StructuredOutputMode == "outputFormat" &&
		options.ResponseFormat != nil && options.ResponseFormat.Type == "json" &&
		len(options.ResponseFormat.Schema) > 0

	var syntheticJSONTool *anthropicTool
	forceJSONToolChoice := false
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		if len(options.ResponseFormat.Schema) > 0 {
			if !useOutputFormat {
				syntheticJSONTool = &anthropicTool{
					Name:        syntheticJSONToolName,
					Description: "Respond with a JSON object.",
					InputSchema: options.ResponseFormat.Schema,
				}
				forceJSONToolChoice = true
			}
		} else {
			// ai-sdk anthropic: ResponseFormat=json without schema → warning,
			// format is ignored (and we still inject the system prompt
			// instruction so the model knows to respond JSON).
			warnings = append(warnings, stream.UnsupportedWarning(
				"responseFormat",
				"JSON response format requires a schema. The response format is ignored.",
			))
			inputMessages = provider.InjectJSONInstruction(inputMessages, nil)
		}
	}

	// Group messages into blocks so consecutive user+tool runs collapse into
	// a single anthropic user message and consecutive assistant runs collapse
	// into a single anthropic assistant message. Mirrors ai-sdk's
	// groupIntoBlocks (convert-to-anthropic-prompt.ts:1088). Anthropic's API
	// requires every tool_result for a given assistant turn to live in one
	// user message immediately following that turn — emitting a separate
	// anthropic message per goai message breaks that pairing.
	var system string
	var systemProviderOpts map[string]any
	var messages []anthropicMessage

	for _, block := range groupIntoBlocks(inputMessages) {
		switch block.Type {
		case blockSystem:
			// goai's legacy behavior: last system message wins (silent
			// overwrite, unlike ai-sdk which throws on multiple).
			for _, msg := range block.Messages {
				system = getTextFromContent(msg.Content)
				systemProviderOpts = msg.ProviderOptions
			}

		case blockUser:
			// Combine all user and tool messages in this block into a
			// single anthropic user message.
			var content []anthropicContentBlock
			for _, msg := range block.Messages {
				switch msg.Role {
				case message.RoleUser:
					content = append(content, convertToAnthropicContent(msg.Content, msg.ProviderOptions)...)
				case message.RoleTool:
					content = append(content, toolMessageToBlocks(msg)...)
				}
			}
			messages = append(messages, anthropicMessage{
				Role:    "user",
				Content: content,
			})

		case blockAssistant:
			// Combine all assistant messages in this block into a single
			// anthropic assistant message.
			var content []anthropicContentBlock
			for _, msg := range block.Messages {
				content = append(content, convertAssistantContent(msg.Content, msg.ProviderOptions)...)
			}
			messages = append(messages, anthropicMessage{
				Role:    "assistant",
				Content: content,
			})
		}
	}

	req := anthropicRequest{
		Model:    modelID,
		Messages: messages,
		Stream:   true,
	}

	if system != "" {
		req.System = system
	}

	if options.Temperature != nil {
		t := *options.Temperature
		if t > 1 {
			warnings = append(warnings, stream.UnsupportedWarning(
				"temperature",
				fmt.Sprintf("%v exceeds anthropic maximum of 1.0. clamped to 1.0", t),
			))
			t = 1
		} else if t < 0 {
			warnings = append(warnings, stream.UnsupportedWarning(
				"temperature",
				fmt.Sprintf("%v is below anthropic minimum of 0. clamped to 0", t),
			))
			t = 0
		}
		req.Temperature = &t
	}
	if options.TopP != nil {
		req.TopP = options.TopP
	}
	if options.TopK != nil {
		req.TopK = options.TopK
	}
	if options.MaxOutputTokens != nil {
		req.MaxTokens = *options.MaxOutputTokens
	} else {
		req.MaxTokens = 4096 // Default
	}
	if len(options.StopSequences) > 0 {
		req.StopSequences = options.StopSequences
	}

	// Tools + hosted-tool beta collection.
	var toolBetas []string
	if len(options.Tools) > 0 {
		var toolWarnings []stream.Warning
		req.Tools, toolBetas, toolWarnings = convertToAnthropicToolsWithWarnings(options.Tools)
		warnings = append(warnings, toolWarnings...)
	}

	// Translate goai's loose ToolChoice (string forms "auto"/"none"/"required"/<tool name>
	// or a structured map) into Anthropic's wire-format object. Mirrors ai-sdk's
	// prepare-tools tool_choice handling: "required" → {type: "any"}, "none" drops
	// both tools[] and tool_choice. Anthropic rejects bare strings on tool_choice.
	choice, dropTools := convertAnthropicToolChoice(options.ToolChoice)
	if dropTools {
		req.Tools = nil
	} else {
		req.ToolChoice = choice
	}

	// Append the synthetic JSON tool after user tools so it doesn't reorder them.
	if syntheticJSONTool != nil {
		req.Tools = append(req.Tools, *syntheticJSONTool)
	}

	// Force the synthetic JSON tool. Overrides any caller-supplied ToolChoice
	// because the caller's intent (structured output) requires it.
	if forceJSONToolChoice {
		req.ToolChoice = map[string]string{"type": "tool", "name": syntheticJSONToolName}
	}

	// thinking - extended thinking configuration
	if opts.Thinking != nil {
		req.Thinking = &anthropicThinking{
			Type:         opts.Thinking.Type,
			BudgetTokens: opts.Thinking.BudgetTokens,
			Display:      opts.Thinking.Display,
		}
	}

	// cacheControl - prompt caching for system message via Message.ProviderOptions
	if systemCC := getCacheControl(systemProviderOpts); systemCC != nil && system != "" {
		req.System = []systemBlock{
			{
				Type:         "text",
				Text:         system,
				CacheControl: systemCC,
			},
		}
	}

	// mcpServers - MCP server configuration
	if len(opts.MCPServers) > 0 {
		servers := make([]anthropicMCPServer, len(opts.MCPServers))
		for i, s := range opts.MCPServers {
			servers[i] = anthropicMCPServer{
				Type:               s.Type,
				Name:               s.Name,
				URL:                s.URL,
				AuthorizationToken: s.AuthorizationToken,
			}
			if s.ToolConfiguration != nil {
				servers[i].ToolConfiguration = &anthropicToolConfiguration{
					Enabled:      s.ToolConfiguration.Enabled,
					AllowedTools: s.ToolConfiguration.AllowedTools,
				}
			}
		}
		req.MCPServers = servers
	}

	// container - Agent Skills configuration
	if opts.Container != nil {
		container := &anthropicContainer{
			ID: opts.Container.ID,
		}
		if len(opts.Container.Skills) > 0 {
			skills := make([]anthropicSkill, len(opts.Container.Skills))
			for i, s := range opts.Container.Skills {
				skills[i] = anthropicSkill{
					Type:    s.Type,
					SkillID: s.SkillID,
					Version: s.Version,
				}
			}
			container.Skills = skills
		}
		req.Container = container
	}

	// contextManagement - server-side context-window compaction policies.
	// Translates camelCase MessagesOptions → snake_case wire format.
	if opts.ContextManagement != nil && len(opts.ContextManagement.Edits) > 0 {
		req.ContextManagement = convertContextManagement(opts.ContextManagement)
	}

	// cacheControl — request-level prompt caching (ai-sdk #17978c6).
	if opts.CacheControl != nil {
		req.CacheControl = &anthropicCacheControl{
			Type: opts.CacheControl.Type,
			TTL:  opts.CacheControl.TTL,
		}
	}

	// metadata.userId → request body metadata.user_id (ai-sdk #05b8ca2).
	if opts.Metadata != nil && opts.Metadata.UserID != "" {
		if req.Metadata == nil {
			req.Metadata = &anthropicMetadata{}
		}
		req.Metadata.UserID = opts.Metadata.UserID
	}

	// speed — fast-mode for Opus 4.6+ (ai-sdk #0a0d29c).
	if opts.Speed != "" {
		req.Speed = opts.Speed
	}

	// inferenceGeo — "us" or "global" (ai-sdk #61f1a61).
	if opts.InferenceGeo != "" {
		req.InferenceGeo = opts.InferenceGeo
	}

	// taskBudget — advisory agentic-turn budget.
	if opts.TaskBudget != nil {
		req.TaskBudget = &anthropicTaskBudget{
			Type:      opts.TaskBudget.Type,
			Total:     opts.TaskBudget.Total,
			Remaining: opts.TaskBudget.Remaining,
		}
	}

	// structuredOutputMode=outputFormat → output_config.format
	// (ai-sdk #d98d9ba renamed this from output_format). Only honored when
	// the configured provider supports native structured output.
	if useOutputFormat {
		name := options.ResponseFormat.Name
		if name == "" {
			name = "response"
		}
		req.OutputConfig = &anthropicOutputConfig{
			Format: &anthropicOutputFormat{
				Type:   "json_schema",
				Name:   name,
				Schema: options.ResponseFormat.Schema,
			},
		}
	}

	// effort → output_config.effort. ai-sdk parity:
	// anthropic-language-model.ts:407-411 — effort is suppressed when
	// thinking.type is explicitly "disabled" (the model can't apply effort
	// when reasoning is off), but otherwise rides alongside any other
	// output_config payload. Merge into req.OutputConfig so the
	// structured-output path above doesn't get clobbered.
	if opts.Effort != "" && (opts.Thinking == nil || opts.Thinking.Type != "disabled") {
		if req.OutputConfig == nil {
			req.OutputConfig = &anthropicOutputConfig{}
		}
		req.OutputConfig.Effort = opts.Effort
	}

	// Collect betas. Order: caller-supplied via providerOptions first so
	// they take precedence over auto-injected ones, then hosted-tool betas,
	// then provider-config tool-type betas. Dedupe preserves first-seen.
	var betas []string
	if len(opts.AnthropicBeta) > 0 {
		betas = append(betas, opts.AnthropicBeta...)
	}
	betas = append(betas, toolBetas...)
	if len(cfg.ToolBetaMap) > 0 {
		for _, t := range req.Tools {
			if typeStr := toolWireType(t); typeStr != "" {
				if beta, ok := cfg.ToolBetaMap[typeStr]; ok {
					betas = append(betas, beta)
				}
			}
		}
	}
	betas = dedupeStrings(betas)

	// Marshal typed request → map so TransformRequestBody can mutate it and
	// per-tool tweaks (strict mode stripping) can be applied uniformly.
	bodyMap, err := structToMap(req)
	if err != nil {
		return nil, betas, warnings, fmt.Errorf("encode request body: %w", err)
	}

	applyToolStrictMode(bodyMap, cfg)

	if cfg.EmitBetasInBody && len(betas) > 0 {
		bodyMap["anthropic_beta"] = toAnyStrings(betas)
	}

	if cfg.TransformRequestBody != nil {
		bodyMap = cfg.TransformRequestBody(bodyMap, betas)
	}

	body, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, betas, warnings, err
	}
	return body, betas, warnings, nil
}

// applyToolStrictMode reconciles Config.ToolsStrict + SupportsStrictTools with
// the assembled tools[] slice. Only function tools (those with an
// input_schema key) get strict; hosted tools are left untouched. When
// SupportsStrictTools is false any pre-existing strict:true is removed.
func applyToolStrictMode(body map[string]any, cfg Config) {
	raw, ok := body["tools"].([]any)
	if !ok || len(raw) == 0 {
		return
	}
	supportsStrict := cfg.supportsStrictTools()
	for _, t := range raw {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		// Function tools have an input_schema field; hosted tools don't.
		if _, isFunctionTool := tm["input_schema"]; !isFunctionTool {
			continue
		}
		if supportsStrict && cfg.ToolsStrict {
			tm["strict"] = true
			continue
		}
		if !supportsStrict {
			delete(tm, "strict")
		}
	}
}

// toolWireType extracts the "type" field from a tool entry for beta-map
// lookups. Hosted tools carry a typed struct; function tools are
// anthropicTool (no Type field). Returns empty if the tool has no Type.
func toolWireType(t any) string {
	raw, err := json.Marshal(t)
	if err != nil {
		return ""
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return ""
	}
	return probe.Type
}

// structToMap round-trips a typed struct through JSON so the builder can
// apply config-driven mutations (TransformRequestBody, strict-mode stripping)
// uniformly on a map[string]any shape.
func structToMap(v any) (map[string]any, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// toAnyStrings converts []string → []any for json.Marshal to emit a JSON
// array of strings (instead of a string-typed slice which encodes the same
// but requires uniform []any for map values).
func toAnyStrings(s []string) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}

// dedupeStrings preserves order while removing duplicates.
func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return in
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

// convertContextManagement maps the user-facing MessagesOptions shape to the
// wire-format anthropicContextManagement. Keep is a union of string "all" and
// {type, value}; we detect that via the Go type of ContextEdit.Keep.
func convertContextManagement(cm *ContextManagement) *anthropicContextManagement {
	edits := make([]anthropicContextEdit, 0, len(cm.Edits))
	for _, e := range cm.Edits {
		edit := anthropicContextEdit{
			Type:                 e.Type,
			ClearToolInputs:      e.ClearToolInputs,
			ExcludeTools:         e.ExcludeTools,
			PauseAfterCompaction: e.PauseAfterCompaction,
			Instructions:         e.Instructions,
		}
		if e.Trigger != nil {
			edit.Trigger = &anthropicContextTrigger{Type: e.Trigger.Type, Value: e.Trigger.Value}
		}
		if e.ClearAtLeast != nil {
			edit.ClearAtLeast = &anthropicContextClearAtLeast{Type: e.ClearAtLeast.Type, Value: e.ClearAtLeast.Value}
		}
		if e.Keep != nil {
			edit.Keep = parseContextKeep(e.Keep)
		}
		edits = append(edits, edit)
	}
	return &anthropicContextManagement{Edits: edits}
}

// parseContextKeep accepts either a string ("all", clear_thinking only) or a
// map/struct with {type, value} — mirrors ai-sdk's discriminated union.
func parseContextKeep(raw any) *anthropicContextKeep {
	switch v := raw.(type) {
	case string:
		if v == "all" {
			return &anthropicContextKeep{All: true}
		}
	case map[string]any:
		k := &anthropicContextKeep{}
		if t, ok := v["type"].(string); ok {
			k.Type = t
		}
		// JSON numbers arrive as float64; int literals stay int.
		switch val := v["value"].(type) {
		case float64:
			k.Value = int(val)
		case int:
			k.Value = val
		case int64:
			k.Value = int(val)
		}
		return k
	case *ContextKeep:
		if v == nil {
			return nil
		}
		if v.All {
			return &anthropicContextKeep{All: true}
		}
		return &anthropicContextKeep{Type: v.Type, Value: v.Value}
	case ContextKeep:
		if v.All {
			return &anthropicContextKeep{All: true}
		}
		return &anthropicContextKeep{Type: v.Type, Value: v.Value}
	}
	return nil
}

func (m *AnthropicModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, jsonToolInjected bool, includeRawChunks bool) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var currentToolCalls = make(map[int]*toolCallAccumulator)
	// Indexes where the synthetic "json" tool's input JSON is being streamed
	// as text. Tracked per-index so mixed real-tool + synthetic streams work.
	syntheticJSONIndexes := make(map[int]bool)
	syntheticJSONSeen := false
	// V3 usage breakdown: collect as individual counts and construct
	// stream.Usage at emit time. All fields are pointers so "unreported"
	// stays nil instead of being a zero-valued int.
	var inputTotal, outputTotal int
	var inputReported, outputReported bool
	var cacheReadInputTokens, cacheCreationInputTokens int
	var outputTokensText, outputTokensReasoning int
	var rawUsage map[string]any
	var finishReason stream.FinishReason
	var contextMgmtMetadata any

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if event.Message != nil && event.Message.Usage != nil {
				u := event.Message.Usage
				inputTotal = u.InputTokens
				inputReported = true
				cacheReadInputTokens = u.CacheReadInputTokens
				cacheCreationInputTokens = u.CacheCreationInputTokens
			}

		case "content_block_start":
			if event.ContentBlock != nil {
				switch event.ContentBlock.Type {
				case "text", "compaction":
					// "compaction" is the block type paired with the
					// compact_20260112 context-management policy (ai-sdk
					// #b094c07): it streams a conversation summary as
					// text, surfaced as the same EventTextStart/Delta
					// sequence as a normal text block.
					if !textStarted {
						textStarted = true
						events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
					}
				case "tool_use":
					// Synthetic JSON tool: surface its input as text so the core's
					// Output.ParseComplete path sees structured JSON as the assistant's
					// message text.
					if jsonToolInjected && event.ContentBlock.Name == syntheticJSONToolName {
						syntheticJSONIndexes[event.Index] = true
						syntheticJSONSeen = true
						if !textStarted {
							textStarted = true
							events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
						}
						break
					}
					acc := &toolCallAccumulator{
						index: event.Index,
						id:    event.ContentBlock.ID,
						name:  event.ContentBlock.Name,
					}
					currentToolCalls[event.Index] = acc
					events <- stream.Event{
						Type: stream.EventToolInputStart,
						Data: stream.ToolInputStartEvent{ID: acc.id, ToolName: acc.name},
					}
				}
			}

		case "content_block_delta":
			if event.Delta != nil {
				switch event.Delta.Type {
				case "text_delta":
					events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: event.Delta.Text}}
				case "compaction_delta":
					// First frame can arrive with content:null (skip
					// silently — ai-sdk #b094c07). Subsequent frames
					// carry the summary text.
					if event.Delta.Content != nil && *event.Delta.Content != "" {
						events <- stream.Event{
							Type: stream.EventTextDelta,
							Data: stream.TextDeltaEvent{Text: *event.Delta.Content},
						}
					}
				case "input_json_delta":
					if syntheticJSONIndexes[event.Index] {
						events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: event.Delta.PartialJSON}}
						break
					}
					if acc, ok := currentToolCalls[event.Index]; ok {
						acc.arguments += event.Delta.PartialJSON
						events <- stream.Event{
							Type: stream.EventToolInputDelta,
							Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: event.Delta.PartialJSON},
						}
					}
				}
			}

		case "content_block_stop":
			if syntheticJSONIndexes[event.Index] {
				delete(syntheticJSONIndexes, event.Index)
				break
			}
			if acc, ok := currentToolCalls[event.Index]; ok {
				events <- stream.Event{
					Type: stream.EventToolInputEnd,
					Data: stream.ToolInputEndEvent{ID: acc.id},
				}

				events <- stream.Event{
					Type: stream.EventToolCall,
					Data: stream.ToolCallEvent{
						ToolCallID: acc.id,
						ToolName:   acc.name,
						Input:      json.RawMessage(acc.arguments),
					},
				}

				// Note: Tool execution is handled by goai.go's executeTools function,
				// not here in the provider. The provider just emits ToolCallEvent.

				delete(currentToolCalls, event.Index)
			}

		case "message_delta":
			if event.Delta != nil {
				finishReason = mapAnthropicStopReason(event.Delta.StopReason)
				if event.Delta.ContextManagement != nil {
					contextMgmtMetadata = mapAppliedContextEdits(event.Delta.ContextManagement)
				}
			}
			if event.Usage != nil {
				outputTotal = event.Usage.OutputTokens
				outputReported = true
				// ai-sdk #b9d105f: cache tokens on streaming, also
				// #2445da4: outputTokens.text breakdown.
				if event.Usage.CacheReadInputTokens > 0 {
					cacheReadInputTokens = event.Usage.CacheReadInputTokens
				}
				if event.Usage.CacheCreationInputTokens > 0 {
					cacheCreationInputTokens = event.Usage.CacheCreationInputTokens
				}
				if event.Usage.OutputTokensByType != nil {
					outputTokensText = event.Usage.OutputTokensByType.Text
					outputTokensReasoning = event.Usage.OutputTokensByType.Reasoning
				}
			}
			// ai-sdk #8c2b1e1: surface full raw usage so callers can
			// read any Anthropic-specific fields we don't model yet.
			if len(event.UsageRaw) > 0 {
				var raw map[string]any
				if err := json.Unmarshal(event.UsageRaw, &raw); err == nil {
					rawUsage = raw
				}
			}

		case "message_stop":
			// Stream ended
		}
	}

	// End text if started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// When the synthetic JSON tool fired, Anthropic reports stop_reason="tool_use"
	// (→ FinishReasonToolCalls). The caller actually wants to treat this as a
	// normal stop so output parsing fires in the core.
	if syntheticJSONSeen && finishReason == stream.FinishReasonToolCalls {
		finishReason = stream.FinishReasonStop
	}

	// Assemble v3 Usage: totals + cache breakdown on InputTokens,
	// text/reasoning split on OutputTokens, raw wire usage on Usage.Raw.
	var usage stream.Usage
	if inputReported {
		usage.InputTokens.Total = stream.IntPtr(inputTotal)
	}
	if outputReported {
		usage.OutputTokens.Total = stream.IntPtr(outputTotal)
	}
	if cacheReadInputTokens > 0 {
		usage.InputTokens.CacheRead = stream.IntPtr(cacheReadInputTokens)
	}
	if cacheCreationInputTokens > 0 {
		usage.InputTokens.CacheWrite = stream.IntPtr(cacheCreationInputTokens)
	}
	if outputTokensText > 0 {
		usage.OutputTokens.Text = stream.IntPtr(outputTokensText)
	}
	if outputTokensReasoning > 0 {
		usage.OutputTokens.Reasoning = stream.IntPtr(outputTokensReasoning)
	}
	if rawUsage != nil {
		usage.Raw = rawUsage
	}

	// providerMetadata.anthropic now carries only Anthropic-specific
	// info that doesn't fit the v3 Usage shape (applied context-
	// management edits). Cache + text/reasoning + raw usage moved onto
	// stream.Usage itself.
	var providerMetadata map[string]any
	if contextMgmtMetadata != nil {
		providerMetadata = map[string]any{
			"anthropic": map[string]any{"contextManagement": contextMgmtMetadata},
		}
	}

	// Emit finish step
	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}

	// Emit finish
	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}
}

// mapAppliedContextEdits converts the wire-format applied_edits into the
// ai-sdk-compatible camelCase shape surfaced via providerMetadata.
func mapAppliedContextEdits(raw *anthropicContextAppliedEdits) any {
	if raw == nil {
		return nil
	}
	out := make([]map[string]any, 0, len(raw.AppliedEdits))
	for _, e := range raw.AppliedEdits {
		entry := map[string]any{"type": e.Type}
		switch e.Type {
		case "clear_tool_uses_20250919":
			entry["clearedToolUses"] = e.ClearedToolUses
			entry["clearedInputTokens"] = e.ClearedInputTokens
		case "clear_thinking_20251015":
			entry["clearedThinkingTurns"] = e.ClearedThinkingTurns
			entry["clearedInputTokens"] = e.ClearedInputTokens
		default:
			// Unknown edit type — still surface it so callers can see what
			// Anthropic cut, but without the typed sub-fields.
			if e.ClearedToolUses > 0 {
				entry["clearedToolUses"] = e.ClearedToolUses
			}
			if e.ClearedThinkingTurns > 0 {
				entry["clearedThinkingTurns"] = e.ClearedThinkingTurns
			}
			if e.ClearedInputTokens > 0 {
				entry["clearedInputTokens"] = e.ClearedInputTokens
			}
		}
		out = append(out, entry)
	}
	return map[string]any{"appliedEdits": out}
}

type toolCallAccumulator struct {
	index     int
	id        string
	name      string
	arguments string
}

func mapAnthropicStopReason(reason string) stream.FinishReason {
	switch reason {
	case "end_turn":
		return stream.FinishReasonStop
	case "max_tokens":
		return stream.FinishReasonLength
	case "stop_sequence":
		return stream.FinishReasonStop
	case "tool_use":
		return stream.FinishReasonToolCalls
	default:
		return stream.FinishReasonOther
	}
}
