package xai

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

// XaiResponsesModel represents an xAI model using the Responses API.
// Used for Grok 4 reasoning-capable models; chat-completion-only models
// (Grok 3 family, grok-code-fast-1) continue to use the openaicompat
// Chat path via Provider.Chat.
type XaiResponsesModel struct {
	id       string
	provider *Provider
	baseURL  string
	apiKey   string
	headers  map[string]string
}

// ID returns the model ID.
func (m *XaiResponsesModel) ID() string { return m.id }

// Provider returns "xai.responses".
func (m *XaiResponsesModel) Provider() string { return "xai.responses" }

// Stream sends a streaming request to xAI using the Responses API.
func (m *XaiResponsesModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)
	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()
	return events, nil
}

func (m *XaiResponsesModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/responses", bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	for k, v := range m.headers {
		req.Header.Set(k, v)
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
			Error: fmt.Errorf("xAI Responses API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func (m *XaiResponsesModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	opts, err := provider.ParseProviderOptions[ResponsesOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on xAI Responses (ai-sdk parity).
	if len(options.StopSequences) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("stopSequences", ""))
	}
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if options.PresencePenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", ""))
	}

	req := responsesRequest{
		Model:  m.id,
		Stream: true,
		Input:  convertToResponsesInput(options.Messages),
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
	if options.ToolChoice != nil {
		choice, choiceWarnings := normalizeResponsesToolChoiceWithWarnings(options.ToolChoice, options.Tools)
		req.ToolChoice = choice
		warnings = append(warnings, choiceWarnings...)
	}

	// reasoningEffort (flat key in providerOptions → reasoning.effort in
	// request). Provider-specific opts.ReasoningEffort wins; otherwise
	// CallOptions.Reasoning lowers into the same wire field (ai-sdk v4
	// reasoning enum).
	effort := opts.ReasoningEffort
	if effort == "" {
		effort = options.Reasoning
	}
	if effort != "" {
		req.Reasoning = &reasoningConfig{Effort: effort}
	}

	// logprobs / topLogprobs. ai-sdk forces logprobs=true when topLogprobs
	// is provided, since xAI returns top-n alternatives via the logprobs
	// channel.
	if opts.TopLogprobs != nil {
		t := true
		req.Logprobs = &t
		req.TopLogprobs = opts.TopLogprobs
	} else if opts.Logprobs != nil {
		req.Logprobs = opts.Logprobs
	}

	// store — xAI defaults to true (opposite of goai's OpenAI default).
	// When store=false, strip item IDs from input and drop reasoning parts
	// that lack encrypted_content (they can't round-trip).
	if opts.Store != nil {
		req.Store = opts.Store
	}

	storeFalse := req.Store != nil && !*req.Store
	if storeFalse {
		for i := range req.Input {
			req.Input[i].ID = ""
		}
		filtered := req.Input[:0]
		for _, item := range req.Input {
			if item.Type == "reasoning" && item.EncryptedContent == "" {
				continue
			}
			filtered = append(filtered, item)
		}
		req.Input = filtered
	}

	// include: merge user-provided include list with the
	// reasoning.encrypted_content default injected when store=false.
	include := append([]string(nil), opts.Include...)
	if storeFalse {
		has := false
		for _, v := range include {
			if v == "reasoning.encrypted_content" {
				has = true
				break
			}
		}
		if !has {
			include = append(include, "reasoning.encrypted_content")
		}
	}
	if len(include) > 0 {
		req.Include = include
	}

	if opts.PreviousResponseID != "" {
		req.PreviousResponseID = opts.PreviousResponseID
	}

	// text.format for json response formats.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		req.Text = &textConfig{}
		if len(options.ResponseFormat.Schema) > 0 {
			name := options.ResponseFormat.Name
			if name == "" {
				name = "response"
			}
			strict := true
			req.Text.Format = &textFormat{
				Type:        "json_schema",
				Name:        name,
				Description: options.ResponseFormat.Description,
				Schema:      options.ResponseFormat.Schema,
				Strict:      &strict,
			}
		} else {
			req.Text.Format = &textFormat{Type: "json_object"}
		}
	}

	if len(options.Tools) > 0 {
		tools, toolWarnings := convertToResponsesToolsWithWarnings(options.Tools)
		req.Tools = tools
		warnings = append(warnings, toolWarnings...)
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

type responsesToolCallAccumulator struct {
	id        string
	name      string
	itemID    string
	arguments string
}

func (m *XaiResponsesModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	// (tools passed through for symmetry with OpenAI parser; unused here
	// because the xAI Responses parser emits tool-call events directly
	// from stream items.)
	_ = tools

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var currentToolCalls = make(map[int]*responsesToolCallAccumulator)
	activeReasoning := make(map[string]bool)
	var usage stream.Usage
	var usageReported bool
	var finishReason stream.FinishReason
	var rawFinishReason string
	var hasFunctionCall bool
	var finishReached bool

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	// Helper: ensure a reasoning-start has been emitted for an item_id
	// before we emit a reasoning-delta or reasoning-end. Covers the
	// encrypted-only reasoning case (#58800f3) where reasoning_summary
	// events never arrive.
	ensureReasoningStart := func(itemID string) {
		if itemID == "" {
			return
		}
		if activeReasoning[itemID] {
			return
		}
		activeReasoning[itemID] = true
		events <- stream.Event{
			Type: stream.EventReasoningStart,
			Data: stream.ReasoningStartEvent{
				ID: itemID,
				ProviderMetadata: map[string]any{
					"xai": map[string]any{"itemId": itemID},
				},
			},
		}
	}

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

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var chunk responsesStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		switch chunk.Type {
		case "response.created", "response.in_progress":
			// Metadata only; no event emitted.

		case "response.output_item.added":
			if chunk.Item == nil {
				continue
			}
			switch chunk.Item.Type {
			case "message":
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
			case "reasoning":
				ensureReasoningStart(chunk.Item.ID)
			case "function_call":
				hasFunctionCall = true
				acc := &responsesToolCallAccumulator{
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

		case "response.reasoning_summary_part.added":
			ensureReasoningStart(chunk.ItemID)

		case "response.reasoning_summary_text.delta":
			if chunk.Delta != "" {
				ensureReasoningStart(chunk.ItemID)
				events <- stream.Event{
					Type: stream.EventReasoningDelta,
					Data: stream.ReasoningDeltaEvent{
						ID:   chunk.ItemID,
						Text: chunk.Delta,
						ProviderMetadata: map[string]any{
							"xai": map[string]any{"itemId": chunk.ItemID},
						},
					},
				}
			}

		case "response.reasoning_summary_text.done":
			// no-op; boundary captured by output_item.done

		case "response.reasoning_text.delta":
			// #8b3e72d: xAI also emits full reasoning-text deltas in
			// addition to summary-text deltas.
			if chunk.Delta != "" {
				ensureReasoningStart(chunk.ItemID)
				events <- stream.Event{
					Type: stream.EventReasoningDelta,
					Data: stream.ReasoningDeltaEvent{
						ID:   chunk.ItemID,
						Text: chunk.Delta,
						ProviderMetadata: map[string]any{
							"xai": map[string]any{"itemId": chunk.ItemID},
						},
					},
				}
			}

		case "response.reasoning_text.done":
			// no-op; boundary captured by output_item.done

		case "response.output_text.delta":
			if chunk.Delta != "" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{
					Type: stream.EventTextDelta,
					Data: stream.TextDeltaEvent{Text: chunk.Delta},
				}
			}

		case "response.output_text.annotation.added":
			// Citations from web_search / x_search hosted tools.
			// Mirrors ai-sdk packages/xai/src/responses/
			// xai-responses-language-model.ts.
			if chunk.Annotation == nil {
				continue
			}
			if src, ok := annotationToSource(*chunk.Annotation); ok {
				events <- stream.Event{Type: stream.EventSource, Data: src}
			}

		case "response.output_text.done":
			// xAI also batches annotations on the text-done event;
			// emit any that arrived. Duplicates with annotation.added
			// are unlikely in practice but harmless — each gets a
			// unique source ID.
			for _, a := range chunk.Annotations {
				if src, ok := annotationToSource(a); ok {
					events <- stream.Event{Type: stream.EventSource, Data: src}
				}
			}

		case "response.function_call_arguments.delta":
			// #902e93b: stream function_call_arguments delta events.
			acc, exists := currentToolCalls[chunk.OutputIndex]
			if exists && chunk.Delta != "" {
				acc.arguments += chunk.Delta
				events <- stream.Event{
					Type: stream.EventToolInputDelta,
					Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: chunk.Delta},
				}
			}

		case "response.function_call_arguments.done":
			// Final arguments are emitted from output_item.done; no-op here.

		case "response.output_item.done":
			if chunk.Item == nil {
				continue
			}
			switch chunk.Item.Type {
			case "message":
				if textStarted {
					textStarted = false
					events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
				}
			case "reasoning":
				// #b937f3e: attach encrypted_content + itemId to providerMetadata.xai.
				// #58800f3: ensure start was emitted even if only encrypted
				// content is present.
				ensureReasoningStart(chunk.Item.ID)
				metadata := map[string]any{}
				xaiMeta := map[string]any{}
				if chunk.Item.ID != "" {
					xaiMeta["itemId"] = chunk.Item.ID
				}
				if chunk.Item.EncryptedContent != "" {
					xaiMeta["reasoningEncryptedContent"] = chunk.Item.EncryptedContent
				}
				if len(xaiMeta) > 0 {
					metadata["xai"] = xaiMeta
				}
				events <- stream.Event{
					Type: stream.EventReasoningEnd,
					Data: stream.ReasoningEndEvent{
						ID:               chunk.Item.ID,
						ProviderMetadata: metadata,
					},
				}
				delete(activeReasoning, chunk.Item.ID)
			case "function_call":
				acc, exists := currentToolCalls[chunk.OutputIndex]
				if !exists {
					continue
				}
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
				delete(currentToolCalls, chunk.OutputIndex)
			}

		case "response.completed", "response.done":
			finishReached = true
			if chunk.Response != nil {
				if chunk.Response.Usage != nil {
					usage = convertXaiResponsesUsage(chunk.Response.Usage)
					usageReported = true
				}
				reason := chunk.Response.Status
				rawFinishReason = reason
				finishReason = mapXaiResponsesFinishReason(reason, hasFunctionCall)
			}

		case "response.incomplete":
			// #c1cc97f: incomplete → use incomplete_details.reason through the
			// normal classifier.
			finishReached = true
			if chunk.Response != nil {
				if chunk.Response.Usage != nil {
					usage = convertXaiResponsesUsage(chunk.Response.Usage)
					usageReported = true
				}
				reason := ""
				if chunk.Response.IncompleteDetails != nil {
					reason = chunk.Response.IncompleteDetails.Reason
				}
				if reason == "" {
					rawFinishReason = "incomplete"
					finishReason = stream.FinishReasonOther
				} else {
					rawFinishReason = reason
					finishReason = mapXaiResponsesFinishReason(reason, hasFunctionCall)
				}
			}

		case "response.failed":
			// #c1cc97f: failed → map like incomplete, but fall back to
			// FinishReasonError when no reason is present.
			finishReached = true
			if chunk.Response != nil {
				if chunk.Response.Usage != nil {
					usage = convertXaiResponsesUsage(chunk.Response.Usage)
					usageReported = true
				}
				reason := ""
				if chunk.Response.IncompleteDetails != nil {
					reason = chunk.Response.IncompleteDetails.Reason
				}
				if reason == "" {
					rawFinishReason = "error"
					finishReason = stream.FinishReasonError
				} else {
					rawFinishReason = reason
					finishReason = mapXaiResponsesFinishReason(reason, hasFunctionCall)
				}
			} else {
				rawFinishReason = "error"
				finishReason = stream.FinishReasonError
			}

		case "error":
			// #72ebb54: mid-stream error surfaces as ErrorEvent without
			// terminating the stream.
			if chunk.Error != nil {
				events <- stream.Event{
					Type: stream.EventError,
					Data: stream.ErrorEvent{
						Error: fmt.Errorf("xAI Responses API error: [%s] %s", chunk.Error.Code, chunk.Error.Message),
					},
				}
			}
		}
	}

	// If we closed a message block but never saw output_item.done, flush it.
	if textStarted {
		textStarted = false
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// #de16a00: if the stream closed without a completed / incomplete /
	// failed event, synthesize a zero-filled Usage and a "other"
	// FinishReason so downstream consumers still see a finish frame.
	if !finishReached {
		if !usageReported {
			usage = stream.UsageFrom(0, 0)
		}
		if finishReason == "" {
			finishReason = stream.FinishReasonOther
		}
	}

	var providerMetadata map[string]any
	if rawFinishReason != "" {
		providerMetadata = map[string]any{
			"xai": map[string]any{"rawFinishReason": rawFinishReason},
		}
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}
}
