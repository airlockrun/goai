package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// ResponsesModel implements the Responses protocol with configurable transport.
type ResponsesModel struct {
	id     string
	config *ResponsesConfig
}

// ResponsesConfig connects the Responses protocol to a provider's endpoint.
// ConfigureRequest runs after per-call headers, so authentication can be applied last.
type ResponsesConfig struct {
	Provider         string
	URL              string
	ConfigureRequest func(*http.Request) error
	HTTPClient       *http.Client
	Headers          map[string]string
	// Generic disables OpenAI model-name capability inference.
	Generic bool
}

// NewResponsesModel creates a reusable Responses protocol model.
func NewResponsesModel(modelID string, config ResponsesConfig) *ResponsesModel {
	if config.Provider == "" || config.URL == "" || config.ConfigureRequest == nil {
		panic("openai: ResponsesConfig requires Provider, URL and ConfigureRequest")
	}
	return &ResponsesModel{id: modelID, config: &config}
}

// ID returns the model ID.
func (m *ResponsesModel) ID() string {
	return m.id
}

// Provider returns the configured provider identity.
func (m *ResponsesModel) Provider() string {
	if m.config != nil {
		return m.config.Provider
	}
	return "openai.responses"
}

// Stream sends a streaming request to OpenAI using the Responses API.
func (m *ResponsesModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *ResponsesModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Build the request
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.config.URL, bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.config.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}
	client := http.DefaultClient
	if err := m.config.ConfigureRequest(req); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if m.config.HTTPClient != nil {
		client = m.config.HTTPClient
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := client.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: m.Provider() + " API request failed", URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: HandleErrorResponse(resp, req.URL.String(), json.RawMessage(reqBody))}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func (m *ResponsesModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	// Parse typed provider options
	opts, err := provider.ParseProviderOptions[ResponsesOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on OpenAI Responses (ai-sdk parity).
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	if options.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}
	if options.PresencePenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", ""))
	}
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if len(options.StopSequences) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("stopSequences", ""))
	}

	caps := GetLanguageModelCapabilities(m.id)
	if m.config != nil && m.config.Generic {
		caps = LanguageModelCapabilities{SystemMessageMode: "system"}
	}
	systemMessageMode := caps.SystemMessageMode
	if opts.ForceReasoning {
		systemMessageMode = "developer"
	}
	if opts.SystemMessageMode != "" {
		systemMessageMode = opts.SystemMessageMode
	}
	if systemMessageMode != "system" && systemMessageMode != "developer" && systemMessageMode != "remove" {
		return nil, warnings, fmt.Errorf("invalid systemMessageMode: %s", systemMessageMode)
	}
	if opts.Conversation != "" && opts.PreviousResponseID != "" {
		return nil, warnings, errors.New("conversation and previousResponseId are mutually exclusive")
	}

	req := responsesRequest{
		Model:              m.id,
		Stream:             true,
		Input:              convertToResponsesInput(options.Messages, systemMessageMode, opts.PassThroughUnsupportedFiles),
		Conversation:       opts.Conversation,
		PreviousResponseID: opts.PreviousResponseID,
		Instructions:       opts.Instructions,
		MaxToolCalls:       opts.MaxToolCalls,
	}

	if options.Temperature != nil {
		req.Temperature = options.Temperature
	}
	if options.TopP != nil {
		req.TopP = options.TopP
	}
	if options.MaxOutputTokens != nil {
		req.MaxOutputTokens = options.MaxOutputTokens
	}

	// Translate goai's loose ToolChoice (bare strings or ai-sdk-shaped objects)
	// into the OpenAI Responses wire form. Bare "auto"/"none"/"required" pass
	// through; {type: "tool", toolName: X} resolves to {type: "function", name: X}
	// for function tools or the hosted-tool wire shape for provider tools.
	// ai-sdk parity: packages/openai/src/responses/openai-responses-prepare-tools.ts.
	if options.ToolChoice != nil {
		req.ToolChoice = convertResponsesToolChoice(options.ToolChoice, options.Tools)
	}

	// Apply provider-specific options from typed struct

	// reasoningEffort - transforms to reasoning.effort in request.
	// Provider-specific opts.ReasoningEffort wins; otherwise the top-level
	// CallOptions.Reasoning lowers into the same wire field (mirrors
	// ai-sdk v4's reasoning enum).
	effort := opts.ReasoningEffort
	if effort == "" {
		effort = options.Reasoning
	}
	if effort != "" {
		req.Reasoning = &reasoningConfig{Effort: effort}
	}
	if (caps.IsReasoningModel || opts.ForceReasoning) && (effort != "none" || !caps.SupportsNonReasoningParameters) {
		if req.Temperature != nil {
			warnings = append(warnings, stream.UnsupportedWarning("temperature", "not supported for reasoning models"))
			req.Temperature = nil
		}
		if req.TopP != nil {
			warnings = append(warnings, stream.UnsupportedWarning("topP", "not supported for reasoning models"))
			req.TopP = nil
		}
	}

	// reasoningSummary
	if opts.ReasoningSummary != "" {
		if req.Reasoning == nil {
			req.Reasoning = &reasoningConfig{}
		}
		req.Reasoning.Summary = opts.ReasoningSummary
	}

	// store option — defaults to false for privacy (don't persist data on OpenAI's servers).
	// When store=false, item IDs are stripped from input since they reference
	// non-persisted items that OpenAI can't look up.
	storeFalse := false
	if opts.Store != nil {
		req.Store = opts.Store
	} else {
		req.Store = &storeFalse
	}

	// Strip item IDs when store=false — OpenAI rejects references to non-persisted items.
	if req.Store != nil && !*req.Store {
		for i := range req.Input {
			req.Input[i].ID = ""
		}
		// ai-sdk #f4a734a: reasoning parts without encrypted_content can't
		// round-trip when store is false (the model's internal reference is
		// never persisted), so OpenAI rejects them. Drop those items
		// defensively. Same class of fix as pivot/fix-goai.md.
		filtered := req.Input[:0]
		for _, item := range req.Input {
			if item.Type == "reasoning" && item.EncryptedContent == "" {
				continue
			}
			filtered = append(filtered, item)
		}
		req.Input = filtered
	}

	// promptCacheKey (session ID for prompt caching)
	if opts.PromptCacheKey != "" {
		req.PromptCacheKey = opts.PromptCacheKey
	}

	// include - extra fields to include in response
	if len(opts.Include) > 0 {
		req.Include = append([]string(nil), opts.Include...)
	}
	if opts.Logprobs != nil {
		var n int
		switch v := opts.Logprobs.(type) {
		case bool:
			if v {
				n = 20
			}
		case int:
			n = v
		case float64:
			n = int(v)
		}
		if n > 0 {
			req.TopLogprobs = &n
			found := false
			for _, field := range req.Include {
				if field == "message.output_text.logprobs" {
					found = true
				}
			}
			if !found {
				req.Include = append(req.Include, "message.output_text.logprobs")
			}
		}
	}
	if (caps.IsReasoningModel || opts.ForceReasoning) && req.Store != nil && !*req.Store {
		found := false
		for _, field := range req.Include {
			if field == "reasoning.encrypted_content" {
				found = true
			}
		}
		if !found {
			req.Include = append(req.Include, "reasoning.encrypted_content")
		}
	}

	// user - unique identifier for end-user
	if opts.User != "" {
		req.User = opts.User
	}

	// parallelToolCalls
	if opts.ParallelToolCalls != nil {
		req.ParallelToolCalls = opts.ParallelToolCalls
	}

	// metadata
	if opts.Metadata != nil {
		req.Metadata = opts.Metadata
	}

	// strictJsonSchema defaults to false to match opencode behavior
	strictJsonSchema := false
	if opts.StrictJsonSchema != nil {
		strictJsonSchema = *opts.StrictJsonSchema
	}

	// text.format and text.verbosity share the same `text` field on the request.
	// Mirrors ai-sdk openai-responses-language-model.ts:282-300.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		if req.Text == nil {
			req.Text = &textConfig{}
		}
		if len(options.ResponseFormat.Schema) > 0 {
			name := options.ResponseFormat.Name
			if name == "" {
				name = "response"
			}
			req.Text.Format = &textFormat{
				Type:        "json_schema",
				Name:        name,
				Description: options.ResponseFormat.Description,
				Schema:      options.ResponseFormat.Schema,
				Strict:      &strictJsonSchema,
			}
		} else {
			req.Text.Format = &textFormat{Type: "json_object"}
		}
	}
	if opts.TextVerbosity != "" {
		if req.Text == nil {
			req.Text = &textConfig{}
		}
		req.Text.Verbosity = opts.TextVerbosity
	}

	// truncation
	if opts.Truncation != "" {
		req.Truncation = opts.Truncation
	}

	// serviceTier
	if opts.ServiceTier != "" {
		req.ServiceTier = opts.ServiceTier
	}

	// safetyIdentifier
	if opts.SafetyIdentifier != "" {
		req.SafetyIdentifier = opts.SafetyIdentifier
	}

	// promptCacheRetention
	if opts.PromptCacheRetention != "" {
		req.PromptCacheRetent = opts.PromptCacheRetention
	}

	// Add tools (already ordered by core)
	if len(options.Tools) > 0 {
		tools, toolWarnings := convertToResponsesToolsWithWarnings(options.Tools, strictJsonSchema)
		req.Tools = tools
		warnings = append(warnings, toolWarnings...)
	}

	// allowedTools restricts the callable subset while keeping the full tools
	// list intact (prompt-caching friendly). It overrides any request-level
	// tool_choice. ai-sdk #15038.
	if opts.AllowedTools != nil {
		req.ToolChoice = allowedToolsChoice(*opts.AllowedTools)
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

// allowedToolsChoice builds the tool_choice: {type: "allowed_tools", ...} wire
// shape. Mode defaults to "auto". ai-sdk #15038.
func allowedToolsChoice(allowed AllowedTools) map[string]any {
	mode := allowed.Mode
	if mode == "" {
		mode = "auto"
	}
	tools := make([]map[string]any, 0, len(allowed.ToolNames))
	for _, name := range allowed.ToolNames {
		tools = append(tools, map[string]any{"type": "function", "name": name})
	}
	return map[string]any{
		"type":  "allowed_tools",
		"mode":  mode,
		"tools": tools,
	}
}

func (m *ResponsesModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	var textStarted bool
	var currentToolCalls = make(map[int]*responsesToolCallAccumulator)
	var currentReasoningID string // Track current reasoning item ID
	var usage stream.Usage
	var responseID, serviceTier string
	var logprobTokens []chatLogprobToken
	var finishReason stream.FinishReason
	// rawFinishReason preserves the provider's original finish-reason
	// string (e.g., "max_tokens", "content_filter") alongside the
	// unified FinishReason enum so callers can surface it to users or
	// distinguish an "error"-classified stop reason from a pure
	// transport/server error (ai-sdk #bcb04df).
	var rawFinishReason string
	var hasFunctionCall bool
	// outputStarted gates early-error handling (ai-sdk #15922): an OpenAI
	// stream can return HTTP 200 and then emit an error/response.failed frame
	// before any model output (e.g. insufficient_quota). Such pre-output
	// errors are surfaced as a terminating error so retry/fallback logic can
	// see a failed stream; errors after real output stay streamed error parts.
	var outputStarted bool
	var terminal bool

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var chunk responsesChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: &goaierrors.JSONParseError{Text: data, Cause: err}}}
			return
		}
		if chunk.Type == "" {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("expected Responses event type", data)}}
			return
		}

		// Mark output as started for any chunk that isn't a lifecycle-only or
		// error frame (ai-sdk #15922 isResponseOutputChunk). Once true, error
		// frames are treated as late (streamed) rather than early (fatal).
		switch chunk.Type {
		case "response.output_item.added", "response.output_item.done", "response.output_text.delta", "response.function_call_arguments.delta", "response.reasoning_summary_text.delta", "response.reasoning_text.delta", "response.output_text.annotation.added":
			outputStarted = true
		}

		switch chunk.Type {
		case "response.created":
			if chunk.Response != nil {
				responseID = chunk.Response.ID
			}

		case "response.output_item.added":
			if chunk.Item == nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("output item event is missing its item", data)}}
				return
			}
			switch chunk.Item.Type {
			case "message":
				// Text output starting. Forward the Responses-API `phase`
				// field (commentary | final_answer, ai-sdk #66a374c) via
				// providerMetadata.openai.phase so callers can preserve it
				// across turns for gpt-5.3-codex and later.
				if !textStarted {
					textStarted = true
					var metadata map[string]any
					if chunk.Item.Phase != "" {
						metadata = map[string]any{
							"openai": map[string]any{"phase": chunk.Item.Phase},
						}
					}
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{ProviderMetadata: metadata}}
				}
			case "reasoning":
				// Reasoning output starting
				currentReasoningID = chunk.Item.ID
				events <- stream.Event{
					Type: stream.EventReasoningStart,
					Data: stream.ReasoningStartEvent{
						ID: currentReasoningID,
					},
				}
			case "function_call":
				hasFunctionCall = true
				acc := &responsesToolCallAccumulator{
					index:     chunk.OutputIndex,
					id:        chunk.Item.CallID,
					name:      chunk.Item.Name,
					itemID:    chunk.Item.ID,
					namespace: chunk.Item.Namespace,
				}
				currentToolCalls[chunk.OutputIndex] = acc
				events <- stream.Event{
					Type: stream.EventToolInputStart,
					Data: stream.ToolInputStartEvent{ID: acc.id, ToolName: acc.name},
				}
			}

		case "response.output_text.delta":
			logprobTokens = append(logprobTokens, chunk.Logprobs...)
			if chunk.Delta != "" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: chunk.Delta}}
			}

		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			if chunk.Delta != "" {
				events <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: currentReasoningID, Text: chunk.Delta}}
			}

		case "response.function_call_arguments.delta":
			acc, exists := currentToolCalls[chunk.OutputIndex]
			if !exists {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("tool argument delta has no matching output item", data)}}
				return
			}
			if exists && chunk.Delta != "" {
				acc.arguments += chunk.Delta
				events <- stream.Event{
					Type: stream.EventToolInputDelta,
					Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: chunk.Delta},
				}
			}

		case "response.output_item.done":
			if chunk.Item == nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("output item event is missing its item", data)}}
				return
			}
			switch chunk.Item.Type {
			case "message":
				// Text output done
				if textStarted {
					textStarted = false
					events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
				}
			case "reasoning":
				// Reasoning output done - capture encrypted content
				// Key names match ai-sdk: itemId, reasoningEncryptedContent
				providerMetadata := make(map[string]any)
				if chunk.Item.EncryptedContent != "" {
					providerMetadata["reasoningEncryptedContent"] = chunk.Item.EncryptedContent
				}
				if chunk.Item.ID != "" {
					providerMetadata["itemId"] = chunk.Item.ID
				}
				events <- stream.Event{
					Type: stream.EventReasoningEnd,
					Data: stream.ReasoningEndEvent{
						ID:               currentReasoningID,
						ProviderMetadata: providerMetadata,
					},
				}
				currentReasoningID = ""
			case "function_call":
				acc, exists := currentToolCalls[chunk.OutputIndex]
				if !exists {
					continue
				}

				// Update with final values
				if chunk.Item.Arguments != "" {
					acc.arguments = chunk.Item.Arguments
				}
				// Namespace may arrive only on the done event (ai-sdk
				// #14789); preserve whichever is non-empty.
				if chunk.Item.Namespace != "" {
					acc.namespace = chunk.Item.Namespace
				}

				events <- stream.Event{
					Type: stream.EventToolInputEnd,
					Data: stream.ToolInputEndEvent{ID: acc.id},
				}

				toolCallMeta := map[string]any{}
				if acc.itemID != "" {
					toolCallMeta["itemId"] = acc.itemID
				}
				if acc.namespace != "" {
					toolCallMeta["namespace"] = acc.namespace
				}
				toolCallEvent := stream.ToolCallEvent{
					ToolCallID: acc.id,
					ToolName:   acc.name,
					Input:      json.RawMessage(acc.arguments),
				}
				if len(toolCallMeta) > 0 {
					toolCallEvent.ProviderMetadata = map[string]any{"openai": toolCallMeta}
				}
				events <- stream.Event{Type: stream.EventToolCall, Data: toolCallEvent}

				// Note: Tool execution is handled by goai.go's executeTools function,
				// not here in the provider. The provider just emits ToolCallEvent.

				delete(currentToolCalls, chunk.OutputIndex)
			}

		case "response.output_text.annotation.added":
			// Citations from web_search / file_search hosted tools.
			// ai-sdk #4f6dc77 enqueues a Source content part per
			// annotation; we mirror that as a SourceEvent.
			if chunk.Annotation == nil {
				continue
			}
			if src, ok := annotationToSource(*chunk.Annotation); ok {
				events <- stream.Event{Type: stream.EventSource, Data: src}
			}

		case "response.completed", "response.incomplete", "response.failed":
			terminal = true
			if chunk.Response == nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("terminal event is missing its response", data)}}
				return
			}
			// A response.failed carrying an error before any output is an
			// early failure: surface it as a terminating error so retry/
			// fallback logic sees a failed stream (ai-sdk #15922).
			if chunk.Type == "response.failed" && chunk.Response.Error != nil {
				e := chunk.Response.Error
				events <- stream.Event{
					Type: stream.EventError,
					Data: stream.ErrorEvent{
						Error: streamAPIError(m.Provider(), e.Code, e.Message, data),
					},
				}
				if !outputStarted {
					return
				}
			}
			if chunk.Response != nil {
				if chunk.Response.ID != "" {
					responseID = chunk.Response.ID
				}
				serviceTier = chunk.Response.ServiceTier
				// Handle usage
				if chunk.Response.Usage != nil {
					usage = stream.UsageFrom(
						chunk.Response.Usage.InputTokens,
						chunk.Response.Usage.OutputTokens,
					)
					u := chunk.Response.Usage
					cached, reasoning := 0, 0
					if u.InputTokensDetails != nil {
						cached = u.InputTokensDetails.CachedTokens
					}
					if u.OutputTokensDetails != nil {
						reasoning = u.OutputTokensDetails.ReasoningTokens
					}
					usage.InputTokens.CacheRead, usage.InputTokens.NoCache = stream.IntPtr(cached), stream.IntPtr(u.InputTokens-cached)
					usage.OutputTokens.Reasoning, usage.OutputTokens.Text = stream.IntPtr(reasoning), stream.IntPtr(u.OutputTokens-reasoning)
				}

				// Handle finish reason. For response.failed (ai-sdk
				// #bcb04df): when incomplete_details.reason is present,
				// map it through the normal finish-reason classifier;
				// otherwise fall back to FinishReasonError. The raw
				// string is preserved on providerMetadata.openai.rawFinishReason
				// so downstream callers can distinguish between, e.g.,
				// a stop-reason-style error and a pure transport error.
				reason := ""
				if chunk.Response.IncompleteDetails != nil {
					reason = chunk.Response.IncompleteDetails.Reason
				}
				if chunk.Type == "response.failed" && reason == "" {
					finishReason = stream.FinishReasonError
					rawFinishReason = "error"
				} else {
					finishReason = mapResponsesFinishReason(reason, hasFunctionCall)
					rawFinishReason = reason
				}
			}
			if chunk.Type == "response.completed" && len(currentToolCalls) > 0 {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: invalidStreamError("response completed with unfinished tool calls", data)}}
				return
			}

		case "error":
			if chunk.Error == nil {
				chunk.Error = &responsesError{Code: chunk.Code, Message: chunk.Message}
			}
			if chunk.Error != nil {
				err := streamAPIError(m.Provider(), chunk.Error.Code, chunk.Error.Message, data)
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
				// A pre-output error frame is fatal: stop the stream so the
				// failure propagates instead of a partial result (ai-sdk #15922).
				if !outputStarted {
					return
				}
				terminal = true
				finishReason = stream.FinishReasonError
			}
		}
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if !terminal {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: incompleteStreamError()}}
		return
	}

	// End text if still started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	metadata := make(map[string]any)
	if rawFinishReason != "" {
		metadata["rawFinishReason"] = rawFinishReason
	}
	if responseID != "" {
		metadata["responseId"] = responseID
	}
	if serviceTier != "" {
		metadata["serviceTier"] = serviceTier
	}
	if len(logprobTokens) > 0 {
		metadata["logprobs"] = mapLogprobTokens(logprobTokens)
	}
	var providerMetadata map[string]any
	if len(metadata) > 0 {
		providerMetadata = map[string]any{"openai": metadata}
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

type responsesToolCallAccumulator struct {
	index     int
	id        string
	name      string
	itemID    string
	namespace string // ai-sdk #14789 — preserved on tool-call providerMetadata
	arguments string
}
