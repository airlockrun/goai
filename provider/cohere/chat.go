package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/provider"
	goairesponse "github.com/airlockrun/goai/response"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// CohereModel represents a Cohere model.
type CohereModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *CohereModel) ID() string {
	return m.id
}

// Provider returns "cohere".
func (m *CohereModel) Provider() string {
	return "cohere"
}

// Stream sends a streaming request to Cohere.
func (m *CohereModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *CohereModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	baseURL := strings.TrimSuffix(strings.TrimRight(m.provider.opts.BaseURL, "/"), "/v1")
	baseURL = strings.TrimSuffix(baseURL, "/v2")
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v2/chat", bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Cohere API request failed", URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Cohere API error: " + string(body), URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			StatusCode: resp.StatusCode, ResponseHeaders: goairesponse.ExtractResponseHeaders(resp), ResponseBody: string(body),
		})}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func (m *CohereModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	// Parse typed provider options
	opts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	messages, documents, err := convertMessages(options.Messages)
	if err != nil {
		return nil, warnings, err
	}
	req := cohereRequest{
		Model: m.id, Messages: messages, Stream: true,
		Documents: documents,
		Seed:      options.Seed, StopSequences: options.StopSequences,
		FrequencyPenalty: options.FrequencyPenalty, PresencePenalty: options.PresencePenalty,
	}

	if options.Temperature != nil {
		req.Temperature = options.Temperature
	}
	if options.TopP != nil {
		req.P = options.TopP
	}
	if options.TopK != nil {
		req.K = options.TopK
	}
	if options.MaxOutputTokens != nil {
		req.MaxTokens = options.MaxOutputTokens
	}

	// Add tools (already ordered by core)
	if len(options.Tools) > 0 {
		req.Tools = convertToCohereTools(options.Tools)
		choice, name := "", ""
		switch value := options.ToolChoice.(type) {
		case nil:
		case string:
			choice = value
		case map[string]any:
			choice, _ = value["type"].(string)
			name, _ = value["toolName"].(string)
		default:
			return nil, warnings, fmt.Errorf("unsupported Cohere tool choice %T", value)
		}
		switch choice {
		case "", "auto":
		case "none":
			req.ToolChoice = "NONE"
		case "required":
			req.ToolChoice = "REQUIRED"
		case "tool":
			req.ToolChoice = "REQUIRED"
			selected := req.Tools[:0]
			for _, t := range req.Tools {
				if t.Function.Name == name {
					selected = append(selected, t)
				}
			}
			if len(selected) == 0 {
				return nil, warnings, fmt.Errorf("Cohere tool %q is not available", name)
			}
			req.Tools = selected
		default:
			return nil, warnings, fmt.Errorf("unsupported Cohere tool choice %q", choice)
		}
	}

	// Apply provider-specific options from typed struct

	// thinking - reasoning configuration
	if opts.Thinking != nil {
		typeName := opts.Thinking.Type
		if typeName == "" {
			typeName = "enabled"
		}
		if typeName != "enabled" && typeName != "disabled" {
			return nil, warnings, fmt.Errorf("invalid Cohere thinking type %q", typeName)
		}
		req.Thinking = &cohereThinking{
			Type:        typeName,
			TokenBudget: opts.Thinking.TokenBudget,
		}
	}

	// ResponseFormat. Cohere always uses type "json_object"; the schema
	// (when present) lives on the same object under json_schema.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		req.ResponseFormat = &cohereResponseFormat{Type: "json_object"}
		if len(options.ResponseFormat.Schema) > 0 {
			req.ResponseFormat.JSONSchema = options.ResponseFormat.Schema
		}
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *CohereModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var reasoningStarted bool
	var usage stream.Usage
	var finishReason stream.FinishReason
	var pending *stream.ToolCallEvent

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
		line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: line}}
		}

		var event cohereStreamEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: &goaierrors.JSONParseError{Text: line, Cause: err}}}
			return
		}

		switch event.Type {
		case "content-start", "content-delta", "tool-plan-delta":
			content := event.Delta.Message.Content
			if content.Type == "thinking" || content.Thinking != "" {
				if !reasoningStarted {
					reasoningStarted = true
					events <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "reasoning-0"}}
				}
				if content.Thinking != "" {
					events <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "reasoning-0", Text: content.Thinking}}
				}
			}
			text := content.Text + event.Delta.Message.ToolPlan
			if text != "" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}}
			}
		case "content-end":
			if reasoningStarted {
				events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning-0"}}
				reasoningStarted = false
			}
			if textStarted {
				events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
				textStarted = false
			}
		case "tool-call-start":
			tc := event.Delta.Message.ToolCalls
			if pending != nil || tc.ID == "" || tc.Function.Name == "" {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: invalid Cohere tool-call-start", goaierrors.ErrInvalidResponse)}}
				return
			}
			pending = &stream.ToolCallEvent{ToolCallID: tc.ID, ToolName: tc.Function.Name, Input: json.RawMessage(tc.Function.Arguments)}
			events <- stream.Event{Type: stream.EventToolInputStart, Data: stream.ToolInputStartEvent{ID: tc.ID, ToolName: tc.Function.Name}}
			if tc.Function.Arguments != "" {
				events <- stream.Event{Type: stream.EventToolInputDelta, Data: stream.ToolInputDeltaEvent{ID: tc.ID, Delta: tc.Function.Arguments}}
			}
		case "tool-call-delta":
			if pending == nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: tool delta without start", goaierrors.ErrInvalidResponse)}}
				return
			}
			args := event.Delta.Message.ToolCalls.Function.Arguments
			pending.Input = append(pending.Input, args...)
			events <- stream.Event{Type: stream.EventToolInputDelta, Data: stream.ToolInputDeltaEvent{ID: pending.ToolCallID, Delta: args}}
		case "tool-call-end":
			if pending == nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: tool end without start", goaierrors.ErrInvalidResponse)}}
				return
			}
			if len(bytes.TrimSpace(pending.Input)) == 0 {
				pending.Input = json.RawMessage(`{}`)
			}
			if !json.Valid(pending.Input) {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: invalid tool arguments", goaierrors.ErrInvalidResponse)}}
				return
			}
			events <- stream.Event{Type: stream.EventToolInputEnd, Data: stream.ToolInputEndEvent{ID: pending.ToolCallID}}
			events <- stream.Event{Type: stream.EventToolCall, Data: *pending}
			pending = nil
		case "message-end":
			if event.Delta.FinishReason != "" {
				finishReason = mapCohereFinishReason(event.Delta.FinishReason)
			}
			var tokens struct {
				Tokens struct {
					Input  int `json:"input_tokens"`
					Output int `json:"output_tokens"`
				} `json:"tokens"`
				Cached int `json:"cached_tokens"`
			}
			if len(event.Delta.Usage) > 0 {
				if err := json.Unmarshal(event.Delta.Usage, &tokens); err != nil {
					events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
					return
				}
				usage = stream.UsageFrom(max(0, tokens.Tokens.Input), max(0, tokens.Tokens.Output))
				cached := min(max(0, tokens.Cached), usage.InputTotal())
				usage.InputTokens.CacheRead = stream.IntPtr(cached)
				usage.InputTokens.NoCache = stream.IntPtr(usage.InputTotal() - cached)
				if err := json.Unmarshal(event.Delta.Usage, &usage.Raw); err != nil {
					events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
					return
				}
			}
		}
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if finishReason == "" || pending != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: incomplete Cohere stream", goaierrors.ErrInvalidResponse)}}
		return
	}
	if reasoningStarted {
		events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning-0"}}
	}

	// End text if started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{
			FinishReason: finishReason,
			Usage:        usage,
		},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{
			FinishReason: finishReason,
			Usage:        usage,
		},
	}
}

func mapCohereFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "COMPLETE":
		return stream.FinishReasonStop
	case "MAX_TOKENS":
		return stream.FinishReasonLength
	case "TOOL_CALL":
		return stream.FinishReasonToolCalls
	default:
		return stream.FinishReasonOther
	}
}
