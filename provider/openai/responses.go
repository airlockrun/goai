package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// ResponsesModel represents an OpenAI model using the Responses API.
// This is the newer API that supports features like reasoning, web search,
// and code interpreter natively.
//
// Note: goai defaults to the Responses API.
// The standard @ai-sdk/openai uses Chat Completions by default
// See provider.go for the default behavior.
type ResponsesModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *ResponsesModel) ID() string {
	return m.id
}

// Provider returns "openai.responses".
func (m *ResponsesModel) Provider() string {
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

	req, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/responses", bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	if m.provider.opts.Organization != "" {
		req.Header.Set("OpenAI-Organization", m.provider.opts.Organization)
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
			Error: fmt.Errorf("OpenAI Responses API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
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

	// Get systemMessageMode from options
	// Default to "system" for non-reasoning models (matching ai-sdk behavior)
	// Reasoning models (o1, o3, o4-mini, gpt-5) should pass "developer" explicitly
	systemMessageMode := "system"
	if opts.SystemMessageMode != "" {
		systemMessageMode = opts.SystemMessageMode
	}

	req := responsesRequest{
		Model:  m.id,
		Stream: true,
		Input:  convertToResponsesInput(options.Messages, systemMessageMode),
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

	// reasoningEffort - transforms to reasoning.effort in request
	if opts.ReasoningEffort != "" {
		req.Reasoning = &reasoningConfig{Effort: opts.ReasoningEffort}
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
		req.Include = opts.Include
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

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *ResponsesModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	var textStarted bool
	var currentToolCalls = make(map[int]*responsesToolCallAccumulator)
	var currentReasoningID string // Track current reasoning item ID
	var usage stream.Usage
	var finishReason stream.FinishReason
	// rawFinishReason preserves the provider's original finish-reason
	// string (e.g., "max_tokens", "content_filter") alongside the
	// unified FinishReason enum so callers can surface it to users or
	// distinguish an "error"-classified stop reason from a pure
	// transport/server error (ai-sdk #bcb04df).
	var rawFinishReason string
	var hasFunctionCall bool

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
		if data == "[DONE]" {
			break
		}

		var chunk responsesChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		switch chunk.Type {
		case "response.created":
			// Response started - metadata available
			if chunk.Response != nil {
				// Could emit metadata event here if needed
			}

		case "response.output_item.added":
			if chunk.Item == nil {
				continue
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
					index:  chunk.OutputIndex,
					id:     chunk.Item.CallID,
					name:   chunk.Item.Name,
					itemID: chunk.Item.ID,
				}
				currentToolCalls[chunk.OutputIndex] = acc
				events <- stream.Event{
					Type: stream.EventToolInputStart,
					Data: stream.ToolInputStartEvent{ID: acc.id, ToolName: acc.name},
				}
			}

		case "response.output_text.delta":
			if chunk.Delta != "" {
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: chunk.Delta}}
			}

		case "response.function_call_arguments.delta":
			acc, exists := currentToolCalls[chunk.OutputIndex]
			if exists && chunk.Delta != "" {
				acc.arguments += chunk.Delta
				events <- stream.Event{
					Type: stream.EventToolInputDelta,
					Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: chunk.Delta},
				}
			}

		case "response.output_item.done":
			if chunk.Item == nil {
				continue
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

				delete(currentToolCalls, chunk.OutputIndex)
			}

		case "response.completed", "response.incomplete", "response.failed":
			if chunk.Response != nil {
				// Handle usage
				if chunk.Response.Usage != nil {
					usage = stream.UsageFrom(
						chunk.Response.Usage.InputTokens,
						chunk.Response.Usage.OutputTokens,
					)
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

		case "error":
			if chunk.Error != nil {
				events <- stream.Event{
					Type: stream.EventError,
					Data: stream.ErrorEvent{
						Error: fmt.Errorf("OpenAI Responses API error: [%s] %s", chunk.Error.Code, chunk.Error.Message),
					},
				}
			}
		}
	}

	// End text if still started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	var providerMetadata map[string]any
	if rawFinishReason != "" {
		providerMetadata = map[string]any{
			"openai": map[string]any{"rawFinishReason": rawFinishReason},
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

type responsesToolCallAccumulator struct {
	index     int
	id        string
	name      string
	itemID    string
	arguments string
}
