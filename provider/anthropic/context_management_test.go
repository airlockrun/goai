package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Tests translated from ai-sdk's anthropic-messages-language-model.test.ts
// Source: ai-sdk/packages/anthropic/src/anthropic-messages-language-model.test.ts
//
// Covered: request body wiring, response metadata extraction (both edit
// types), and the "all" string literal union for clear_thinking_20251015.

func TestAnthropicModel_ContextManagement_Request(t *testing.T) {
	t.Run("should send clear_tool_uses edit with all options in request body", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","usage":{"input_tokens":10,"output_tokens":0}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
				`{"type":"message_stop"}`,
			}
			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		m := createTestProvider(server.URL).Model("claude-3-haiku-20240307")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hello")},
			ProviderOptions: map[string]any{
				"contextManagement": map[string]any{
					"edits": []map[string]any{
						{
							"type":    "clear_tool_uses_20250919",
							"trigger": map[string]any{"type": "input_tokens", "value": 50000},
							"keep":    map[string]any{"type": "tool_uses", "value": 5},
							"clearAtLeast": map[string]any{
								"type":  "input_tokens",
								"value": 10000,
							},
							"clearToolInputs": true,
							"excludeTools":    []any{"important_tool"},
						},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		for range events {
		}

		cm, ok := receivedBody["context_management"].(map[string]any)
		if !ok {
			t.Fatal("expected context_management object in request body")
		}
		edits, ok := cm["edits"].([]any)
		if !ok || len(edits) != 1 {
			t.Fatalf("expected 1 edit, got %v", cm["edits"])
		}
		edit := edits[0].(map[string]any)
		if edit["type"] != "clear_tool_uses_20250919" {
			t.Errorf("edit type = %v, want clear_tool_uses_20250919", edit["type"])
		}
		trigger := edit["trigger"].(map[string]any)
		if trigger["type"] != "input_tokens" || trigger["value"].(float64) != 50000 {
			t.Errorf("trigger = %v, want input_tokens/50000", trigger)
		}
		keep := edit["keep"].(map[string]any)
		if keep["type"] != "tool_uses" || keep["value"].(float64) != 5 {
			t.Errorf("keep = %v, want tool_uses/5", keep)
		}
		// clear_at_least (snake_case on wire)
		cal := edit["clear_at_least"].(map[string]any)
		if cal["type"] != "input_tokens" || cal["value"].(float64) != 10000 {
			t.Errorf("clear_at_least = %v, want input_tokens/10000", cal)
		}
		if edit["clear_tool_inputs"] != true {
			t.Errorf("clear_tool_inputs = %v, want true", edit["clear_tool_inputs"])
		}
		exclude := edit["exclude_tools"].([]any)
		if len(exclude) != 1 || exclude[0] != "important_tool" {
			t.Errorf("exclude_tools = %v, want [important_tool]", exclude)
		}
	})

	t.Run("should serialize clear_thinking keep=all as bare string", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n"))
			w.Write([]byte(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n"))
			w.Write([]byte(`data: {"type":"message_stop"}` + "\n\n"))
		}))
		defer server.Close()

		m := createTestProvider(server.URL).Model("claude-3-haiku-20240307")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hello")},
			ProviderOptions: map[string]any{
				"contextManagement": map[string]any{
					"edits": []map[string]any{
						{
							"type": "clear_thinking_20251015",
							"keep": "all",
						},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		for range events {
		}

		edits := receivedBody["context_management"].(map[string]any)["edits"].([]any)
		keep := edits[0].(map[string]any)["keep"]
		if keep != "all" {
			t.Errorf("keep = %v, want \"all\"", keep)
		}
	})
}

func TestAnthropicModel_ContextManagement_ResponseMetadata(t *testing.T) {
	t.Run("should surface applied_edits via providerMetadata.anthropic.contextManagement", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":100,"output_tokens":0}}}` + "\n\n"))
			// message_delta carries the applied context_management edits.
			w.Write([]byte(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":5,"cleared_input_tokens":10000},{"type":"clear_thinking_20251015","cleared_thinking_turns":2,"cleared_input_tokens":4200}]}},"usage":{"output_tokens":50}}` + "\n\n"))
			w.Write([]byte(`data: {"type":"message_stop"}` + "\n\n"))
		}))
		defer server.Close()

		m := createTestProvider(server.URL).Model("claude-3-haiku-20240307")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}

		var finishMetadata map[string]any
		for ev := range events {
			if ev.Type == stream.EventFinish {
				finishMetadata = ev.Data.(stream.FinishEvent).ProviderMetadata
			}
		}
		if finishMetadata == nil {
			t.Fatal("FinishEvent.ProviderMetadata was nil")
		}
		anth, ok := finishMetadata["anthropic"].(map[string]any)
		if !ok {
			t.Fatalf("providerMetadata.anthropic missing: %v", finishMetadata)
		}
		cm, ok := anth["contextManagement"].(map[string]any)
		if !ok {
			t.Fatalf("anthropic.contextManagement missing: %v", anth)
		}
		applied, ok := cm["appliedEdits"].([]map[string]any)
		if !ok {
			t.Fatalf("appliedEdits wrong shape: %T %v", cm["appliedEdits"], cm["appliedEdits"])
		}
		if len(applied) != 2 {
			t.Fatalf("appliedEdits len = %d, want 2", len(applied))
		}
		if applied[0]["type"] != "clear_tool_uses_20250919" {
			t.Errorf("applied[0].type = %v", applied[0]["type"])
		}
		if applied[0]["clearedToolUses"] != 5 {
			t.Errorf("applied[0].clearedToolUses = %v, want 5", applied[0]["clearedToolUses"])
		}
		if applied[0]["clearedInputTokens"] != 10000 {
			t.Errorf("applied[0].clearedInputTokens = %v, want 10000", applied[0]["clearedInputTokens"])
		}
		if applied[1]["type"] != "clear_thinking_20251015" {
			t.Errorf("applied[1].type = %v", applied[1]["type"])
		}
		if applied[1]["clearedThinkingTurns"] != 2 {
			t.Errorf("applied[1].clearedThinkingTurns = %v, want 2", applied[1]["clearedThinkingTurns"])
		}
	})
}
