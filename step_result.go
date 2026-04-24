// Package goai provides a Go implementation of AI SDK functionality.
package goai

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// StepResult represents the result of a single step in the generation process.
// Equivalent to ai-sdk's StepResult type.
type StepResult struct {
	// Content contains all content parts generated in this step.
	Content []ContentPart

	// FinishReason indicates why this step ended.
	FinishReason stream.FinishReason

	// Usage contains token usage for this step.
	Usage stream.Usage

	// Response contains response metadata including messages.
	Response StepResponseMeta
}

// StepResponseMeta contains response metadata for a step.
type StepResponseMeta struct {
	// Messages are the response messages generated in this step.
	Messages []message.Message
}

// ContentPart is the interface for all content part types.
type ContentPart interface {
	contentPartType() string
}

// TextContentPart represents generated text.
type TextContentPart struct {
	Text string
}

func (TextContentPart) contentPartType() string { return "text" }

// ReasoningContentPart represents reasoning/thinking text.
type ReasoningContentPart struct {
	ID              string
	Text            string
	ProviderOptions map[string]any // Contains provider-specific data like "reasoningEncryptedContent"
}

func (ReasoningContentPart) contentPartType() string { return "reasoning" }

// ToolCallContentPart represents a tool call.
type ToolCallContentPart struct {
	stream.ToolCall
	Dynamic bool
}

func (ToolCallContentPart) contentPartType() string { return "tool-call" }

// ToolResultContentPart represents a tool result.
type ToolResultContentPart struct {
	stream.ToolResultEvent
	Dynamic bool
}

func (ToolResultContentPart) contentPartType() string { return "tool-result" }

// Text returns the concatenated text from all text content parts.
func (s *StepResult) Text() string {
	var result string
	for _, part := range s.Content {
		if textPart, ok := part.(TextContentPart); ok {
			result += textPart.Text
		}
	}
	return result
}

// Reasoning returns all reasoning content parts.
func (s *StepResult) Reasoning() []ReasoningContentPart {
	var result []ReasoningContentPart
	for _, part := range s.Content {
		if reasoningPart, ok := part.(ReasoningContentPart); ok {
			result = append(result, reasoningPart)
		}
	}
	return result
}

// ReasoningText returns the concatenated text from all reasoning parts.
func (s *StepResult) ReasoningText() string {
	reasoning := s.Reasoning()
	if len(reasoning) == 0 {
		return ""
	}
	var result string
	for _, part := range reasoning {
		result += part.Text
	}
	return result
}

// ToolCalls returns all tool call content parts.
func (s *StepResult) ToolCalls() []stream.ToolCall {
	var result []stream.ToolCall
	for _, part := range s.Content {
		if tcPart, ok := part.(ToolCallContentPart); ok {
			result = append(result, tcPart.ToolCall)
		}
	}
	return result
}

// StaticToolCalls returns tool calls that are not dynamic.
func (s *StepResult) StaticToolCalls() []stream.ToolCall {
	var result []stream.ToolCall
	for _, part := range s.Content {
		if tcPart, ok := part.(ToolCallContentPart); ok && !tcPart.Dynamic {
			result = append(result, tcPart.ToolCall)
		}
	}
	return result
}

// DynamicToolCalls returns tool calls that are dynamic.
func (s *StepResult) DynamicToolCalls() []stream.ToolCall {
	var result []stream.ToolCall
	for _, part := range s.Content {
		if tcPart, ok := part.(ToolCallContentPart); ok && tcPart.Dynamic {
			result = append(result, tcPart.ToolCall)
		}
	}
	return result
}

// ToolResults returns all tool result content parts.
func (s *StepResult) ToolResults() []stream.ToolResultEvent {
	var result []stream.ToolResultEvent
	for _, part := range s.Content {
		if trPart, ok := part.(ToolResultContentPart); ok {
			result = append(result, trPart.ToolResultEvent)
		}
	}
	return result
}

// StaticToolResults returns tool results that are not dynamic.
func (s *StepResult) StaticToolResults() []stream.ToolResultEvent {
	var result []stream.ToolResultEvent
	for _, part := range s.Content {
		if trPart, ok := part.(ToolResultContentPart); ok && !trPart.Dynamic {
			result = append(result, trPart.ToolResultEvent)
		}
	}
	return result
}

// DynamicToolResults returns tool results that are dynamic.
func (s *StepResult) DynamicToolResults() []stream.ToolResultEvent {
	var result []stream.ToolResultEvent
	for _, part := range s.Content {
		if trPart, ok := part.(ToolResultContentPart); ok && trPart.Dynamic {
			result = append(result, trPart.ToolResultEvent)
		}
	}
	return result
}

// GetFinishReason returns the finish reason (implements StepResultData interface).
func (s *StepResult) GetFinishReason() stream.FinishReason {
	return s.FinishReason
}

// GetUsage returns the usage statistics (implements StepResultData interface).
func (s *StepResult) GetUsage() stream.Usage {
	return s.Usage
}
