package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
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

	req, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/chat", bytes.NewReader(reqBody))
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
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{
			Error: fmt.Errorf("Cohere API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func (m *CohereModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	// Parse typed provider options
	opts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on Cohere (per plan inventory): seed.
	// ai-sdk's cohere model itself doesn't flag these in getArgs, but
	// Cohere's v1 Chat request has no seed field, so surface a warning
	// when the caller provides one so the silent drop is visible.
	if options.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}

	var preamble string
	var chatHistory []cohereMessage
	var userMessage string

	for i, msg := range options.Messages {
		switch msg.Role {
		case message.RoleSystem:
			preamble = getTextFromContent(msg.Content)
		case message.RoleUser:
			text := getTextFromContent(msg.Content)
			// Last user message becomes the query
			if i == len(options.Messages)-1 {
				userMessage = text
			} else {
				chatHistory = append(chatHistory, cohereMessage{
					Role:    "USER",
					Message: text,
				})
			}
		case message.RoleAssistant:
			chatHistory = append(chatHistory, cohereMessage{
				Role:    "CHATBOT",
				Message: getTextFromContent(msg.Content),
			})
		case message.RoleTool:
			for _, part := range msg.Content.Parts {
				if tr, ok := part.(message.ToolResultPart); ok {
					resultStr := ""
					switch v := tr.Result.(type) {
					case string:
						resultStr = v
					default:
						if b, err := json.Marshal(v); err == nil {
							resultStr = string(b)
						}
					}
					chatHistory = append(chatHistory, cohereMessage{
						Role:    "TOOL",
						Message: resultStr,
					})
				}
			}
		}
	}

	req := cohereRequest{
		Model:       m.id,
		Message:     userMessage,
		ChatHistory: chatHistory,
		Stream:      true,
	}

	if preamble != "" {
		req.Preamble = preamble
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
	}

	// Apply provider-specific options from typed struct

	// thinking - reasoning configuration
	if opts.Thinking != nil {
		req.Thinking = &cohereThinking{
			Type:        opts.Thinking.Type,
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

func (m *CohereModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var usage stream.Usage
	var finishReason stream.FinishReason
	var pendingToolCalls []stream.ToolCallEvent

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if line == "" {
			continue
		}

		var event cohereStreamEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		switch event.EventType {
		case "text-generation":
			if event.Text != "" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: event.Text}}
			}

		case "tool-calls-generation":
			for _, tc := range event.ToolCalls {
				inputBytes, _ := json.Marshal(tc.Parameters)
				pendingToolCalls = append(pendingToolCalls, stream.ToolCallEvent{
					ToolCallID: tc.Name, // Cohere uses name as ID
					ToolName:   tc.Name,
					Input:      inputBytes,
				})
			}

		case "stream-end":
			finishReason = mapCohereFinishReason(event.FinishReason)
			if event.Response != nil && event.Response.Meta != nil {
				if event.Response.Meta.Tokens != nil {
					usage = stream.UsageFrom(
						event.Response.Meta.Tokens.InputTokens,
						event.Response.Meta.Tokens.OutputTokens,
					)
				}
			}
		}
	}

	// End text if started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// Process tool calls
	for _, tc := range pendingToolCalls {
		events <- stream.Event{
			Type: stream.EventToolInputStart,
			Data: stream.ToolInputStartEvent{ID: tc.ToolCallID, ToolName: tc.ToolName},
		}
		events <- stream.Event{
			Type: stream.EventToolInputEnd,
			Data: stream.ToolInputEndEvent{ID: tc.ToolCallID},
		}
		events <- stream.Event{Type: stream.EventToolCall, Data: tc}

		// Note: Tool execution is handled by goai.go's executeTools function,
		// not here in the provider. The provider just emits ToolCallEvent.
	}

	if len(pendingToolCalls) > 0 && finishReason == "" {
		finishReason = stream.FinishReasonToolCalls
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
