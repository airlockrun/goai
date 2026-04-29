// Package testutil provides test utilities for goai.
// Source: ai-sdk/packages/ai/src/test/mock-language-model-v3.ts
package testutil

import (
	"context"
	"encoding/json"

	"github.com/airlockrun/goai/stream"
)

// MockLanguageModel is a mock implementation of LanguageModel for testing.
// It mirrors ai-sdk's MockLanguageModelV4.
type MockLanguageModel struct {
	id       string
	provider string

	// DoStreamFunc is called for each Stream call. If set, takes precedence over StreamResponses.
	DoStreamFunc func(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error)

	// StreamResponses is a list of responses to return in sequence.
	// Each call to Stream returns the next response.
	StreamResponses [][]stream.Event

	// DoStreamCalls records all calls to Stream for assertions.
	DoStreamCalls []*stream.CallOptions

	streamCallCount int
}

// MockLanguageModelOptions configures the mock language model.
type MockLanguageModelOptions struct {
	// ID is the model identifier. Defaults to "mock-model-id".
	ID string

	// Provider is the provider identifier. Defaults to "mock-provider".
	Provider string

	// DoStreamFunc is called for each Stream call.
	// If set, takes precedence over StreamResponse/StreamResponses.
	DoStreamFunc func(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error)

	// StreamResponse is a single response to return for all Stream calls.
	// Use StreamResponses for multiple sequential responses.
	StreamResponse []stream.Event

	// StreamResponses is a list of responses to return in sequence.
	// Takes precedence over StreamResponse.
	StreamResponses [][]stream.Event
}

// NewMockLanguageModel creates a new mock language model.
func NewMockLanguageModel(opts MockLanguageModelOptions) *MockLanguageModel {
	m := &MockLanguageModel{
		id:           opts.ID,
		provider:     opts.Provider,
		DoStreamFunc: opts.DoStreamFunc,
	}

	if m.id == "" {
		m.id = "mock-model-id"
	}
	if m.provider == "" {
		m.provider = "mock-provider"
	}

	// Handle StreamResponse vs StreamResponses
	if opts.StreamResponses != nil {
		m.StreamResponses = opts.StreamResponses
	} else if opts.StreamResponse != nil {
		m.StreamResponses = [][]stream.Event{opts.StreamResponse}
	}

	return m
}

func (m *MockLanguageModel) ID() string       { return m.id }
func (m *MockLanguageModel) Provider() string { return m.provider }

func (m *MockLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	m.DoStreamCalls = append(m.DoStreamCalls, options)

	if m.DoStreamFunc != nil {
		return m.DoStreamFunc(ctx, options)
	}

	// Get the response for this call
	var response []stream.Event
	if len(m.StreamResponses) > 0 {
		idx := m.streamCallCount
		if idx >= len(m.StreamResponses) {
			idx = len(m.StreamResponses) - 1 // Repeat last response
		}
		response = m.StreamResponses[idx]
	}
	m.streamCallCount++

	events := make(chan stream.Event, len(response)+1)
	go func() {
		defer close(events)
		for _, event := range response {
			select {
			case <-ctx.Done():
				return
			case events <- event:
			}
		}
	}()

	return events, nil
}

// Helper functions for creating mock events

// MockTextResponse creates events for a simple text response.
func MockTextResponse(text string, usage stream.Usage) []stream.Event {
	return []stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason: stream.FinishReasonStop,
			Usage:        usage,
		}},
	}
}

// MockStreamedTextResponse creates events for a streamed text response with multiple chunks.
func MockStreamedTextResponse(chunks []string, usage stream.Usage) []stream.Event {
	events := []stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
	}
	for _, chunk := range chunks {
		events = append(events, stream.Event{
			Type: stream.EventTextDelta,
			Data: stream.TextDeltaEvent{Text: chunk},
		})
	}
	events = append(events,
		stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason: stream.FinishReasonStop,
			Usage:        usage,
		}},
	)
	return events
}

// MockToolCallResponse creates events for a tool call response.
func MockToolCallResponse(toolCallID, toolName string, input any, usage stream.Usage) []stream.Event {
	inputJSON, _ := json.Marshal(input)
	return []stream.Event{
		{Type: stream.EventToolCall, Data: stream.ToolCallEvent{
			ToolCallID: toolCallID,
			ToolName:   toolName,
			Input:      inputJSON,
		}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason: stream.FinishReasonToolCalls,
			Usage:        usage,
		}},
	}
}

// MockTextWithToolCallResponse creates events for a response with both text and tool call.
func MockTextWithToolCallResponse(text, toolCallID, toolName string, input any, usage stream.Usage) []stream.Event {
	inputJSON, _ := json.Marshal(input)
	return []stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventToolCall, Data: stream.ToolCallEvent{
			ToolCallID: toolCallID,
			ToolName:   toolName,
			Input:      inputJSON,
		}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason: stream.FinishReasonToolCalls,
			Usage:        usage,
		}},
	}
}

// MockErrorResponse creates events for an error response.
func MockErrorResponse(err error) []stream.Event {
	return []stream.Event{
		{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}},
	}
}

// MockUsage creates a Usage struct with common defaults.
func MockUsage(promptTokens, completionTokens int) stream.Usage {
	return stream.UsageFrom(promptTokens, completionTokens)
}

// MockReasoningResponse creates events for a response with reasoning.
func MockReasoningResponse(reasoningText, text string, usage stream.Usage) []stream.Event {
	return []stream.Event{
		{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "r0"}},
		{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "r0", Text: reasoningText}},
		{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "r0"}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason: stream.FinishReasonStop,
			Usage:        usage,
		}},
	}
}
