// Package testutil provides deterministic offline models and event fixtures.
package testutil

import (
	"encoding/json"

	"github.com/airlockrun/goai/stream"
)

// MockTextResponse creates events for a simple text response.
func MockTextResponse(text string, usage stream.Usage) []stream.Event {
	return MockStreamedTextResponse([]string{text}, usage)
}

// MockStreamedTextResponse creates events for a text response with multiple chunks.
func MockStreamedTextResponse(chunks []string, usage stream.Usage) []stream.Event {
	events := []stream.Event{{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}}
	for _, chunk := range chunks {
		events = append(events, stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: chunk}})
	}
	return append(events,
		stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop, Usage: usage}},
	)
}

// MockToolCallResponse creates events for a tool call response.
// Input must be JSON-serializable.
func MockToolCallResponse(toolCallID, toolName string, input any, usage stream.Usage) []stream.Event {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		panic(err)
	}
	return []stream.Event{
		{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: toolCallID, ToolName: toolName, Input: inputJSON}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonToolCalls, Usage: usage}},
	}
}

// MockTextWithToolCallResponse creates events containing text and a tool call.
func MockTextWithToolCallResponse(text, toolCallID, toolName string, input any, usage stream.Usage) []stream.Event {
	events := MockStreamedTextResponse([]string{text}, usage)
	return append(events[:len(events)-1], MockToolCallResponse(toolCallID, toolName, input, usage)...)
}

// MockErrorResponse creates events for an error response.
func MockErrorResponse(err error) []stream.Event {
	return []stream.Event{{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}}
}

// MockUsage creates explicit input and output token usage.
func MockUsage(promptTokens, completionTokens int) stream.Usage {
	return stream.UsageFrom(promptTokens, completionTokens)
}

// MockReasoningResponse creates events for a response with reasoning.
func MockReasoningResponse(reasoningText, text string, usage stream.Usage) []stream.Event {
	events := []stream.Event{
		{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "r0"}},
		{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "r0", Text: reasoningText}},
		{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "r0"}},
	}
	return append(events, MockTextResponse(text, usage)...)
}
