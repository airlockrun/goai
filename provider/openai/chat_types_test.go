package openai

import (
	"encoding/json"
	"strings"
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
							Output:     message.TextOutput{Value: "72°F"},
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
							Output:     message.TextOutput{Value: "screenshot taken"},
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
							Output:     message.TextOutput{Value: "contents"},
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

// TestConvertToChatMessages_OutputVariantWire asserts every
// ToolResultOutput variant stringifies via ToolOutputWire into the tool
// message Content, and that there is no is_error field on the OpenAI chat
// message (ADR §8 — Chat Completions has no error flag).
func TestConvertToChatMessages_OutputVariantWire(t *testing.T) {
	tests := []struct {
		name   string
		output message.ToolResultOutput
	}{
		{"text", message.TextOutput{Value: "plain text"}},
		{"json", message.JSONOutput{Value: map[string]any{"k": "v"}}},
		{"error-text", message.ErrorTextOutput{Value: "boom"}},
		{"error-json", message.ErrorJSONOutput{Value: map[string]any{"code": 1}}},
		{"execution-denied", message.ExecutionDeniedOutput{Reason: "nope"}},
		{"content", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "hi"}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgs := []message.Message{{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "c1", ToolName: "t", Output: tt.output},
				}},
			}}
			result, err := convertToChatMessages(msgs)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(result) != 1 {
				t.Fatalf("expected 1 message, got %d", len(result))
			}
			want := message.ToolOutputWire(tt.output)
			if result[0].Content != want {
				t.Errorf("Content = %v, want %q", result[0].Content, want)
			}
			// No is_error field exists on the OpenAI chat message: the
			// marshaled JSON must never carry one.
			raw, err := json.Marshal(result[0])
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(raw), "is_error") {
				t.Errorf("openai tool message must not contain is_error: %s", raw)
			}
		})
	}
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
