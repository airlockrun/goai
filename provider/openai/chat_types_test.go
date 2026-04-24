package openai

import (
	"testing"

	"github.com/airlockrun/goai/message"
)

// Tests for chat message conversion - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/chat/convert-to-openai-chat-messages.test.ts

func TestConvertToChatMessages_ToolResults(t *testing.T) {
	t.Run("should convert simple tool result", func(t *testing.T) {
		msgs := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "get_weather",
							Result:     "72°F",
						},
					},
				},
			},
		}

		result, err := convertToChatMessages(msgs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		if result[0].Role != "tool" {
			t.Errorf("expected role 'tool', got '%s'", result[0].Role)
		}
		if result[0].Content != "72°F" {
			t.Errorf("expected content '72°F', got '%v'", result[0].Content)
		}
		if result[0].ToolCallID != "call_123" {
			t.Errorf("expected tool_call_id 'call_123', got '%s'", result[0].ToolCallID)
		}
	})

	t.Run("should error on image part in tool result", func(t *testing.T) {
		msgs := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "screenshot",
							Result:     "screenshot taken",
						},
						message.ImagePart{
							Image:    "base64data",
							MimeType: "image/png",
						},
					},
				},
			},
		}

		_, err := convertToChatMessages(msgs)
		if err == nil {
			t.Fatal("expected error for image part in tool result")
		}
	})

	t.Run("should error on file part in tool result", func(t *testing.T) {
		msgs := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_456",
							ToolName:   "read_pdf",
							Result:     "contents",
						},
						message.FilePart{
							Data:     "JVBERi0=",
							MimeType: "application/pdf",
						},
					},
				},
			},
		}

		_, err := convertToChatMessages(msgs)
		if err == nil {
			t.Fatal("expected error for file part in tool result")
		}
	})
}

// Regression for ai-sdk #953385d: empty tool-call input serializes as
// "{}" so Chat Completions never sees an empty arguments string.
func TestConvertToChatMessages_EmptyToolCallArgsDefaultsToObject(t *testing.T) {
	msgs := []message.Message{
		{
			Role: message.RoleAssistant,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolCallPart{ID: "call_1", Name: "probe", Input: nil},
				},
			},
		},
	}
	got, err := convertToChatMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(got[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 assistant message with 1 tool call, got %+v", got)
	}
	if got[0].ToolCalls[0].Function.Arguments != "{}" {
		t.Errorf("Arguments = %q, want \"{}\"", got[0].ToolCalls[0].Function.Arguments)
	}
}
