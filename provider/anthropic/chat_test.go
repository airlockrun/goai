package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/anthropic/src/anthropic-messages-language-model.test.ts

func createTestProvider(serverURL string) *Provider {
	return New(Options{
		APIKey:  "test-api-key",
		BaseURL: serverURL,
	})
}

func TestAnthropicModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.Model("claude-3-haiku-20240307")

	if model.ID() != "claude-3-haiku-20240307" {
		t.Errorf("expected model ID claude-3-haiku-20240307, got %s", model.ID())
	}
}

func TestAnthropicModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.Model("claude-3-haiku-20240307")

	if model.Provider() != "anthropic" {
		t.Errorf("expected provider anthropic, got %s", model.Provider())
	}
}

func TestAnthropicModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", "}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"World!"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`,
				`{"type":"message_stop"}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var textParts []string
		for event := range events {
			if event.Type == stream.EventTextDelta {
				if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
					textParts = append(textParts, delta.Text)
				}
			}
		}

		text := strings.Join(textParts, "")
		if text != "Hello, World!" {
			t.Errorf("expected text 'Hello, World!', got %s", text)
		}
	})

	t.Run("should extract usage", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":0}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":30}}`,
				`{"type":"message_stop"}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var usage stream.Usage
		for event := range events {
			if event.Type == stream.EventFinish {
				if finish, ok := event.Data.(stream.FinishEvent); ok {
					usage = finish.Usage
				}
			}
		}

		if usage.InputTotal() != 4 {
			t.Errorf("expected prompt tokens 4, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 30 {
			t.Errorf("expected completion tokens 30, got %d", usage.OutputTotal())
		}
	})
}

func TestAnthropicModel_RequestBody(t *testing.T) {
	t.Run("should send the model id and settings", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"output_tokens":0}}}`,
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		temp := 0.5
		topP := 0.9
		topK := 10
		maxTokens := 1000

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: &maxTokens,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if receivedBody["model"] != "claude-3-haiku-20240307" {
			t.Errorf("expected model claude-3-haiku-20240307, got %v", receivedBody["model"])
		}
		if receivedBody["temperature"] != 0.5 {
			t.Errorf("expected temperature 0.5, got %v", receivedBody["temperature"])
		}
		if receivedBody["top_p"] != 0.9 {
			t.Errorf("expected top_p 0.9, got %v", receivedBody["top_p"])
		}
		if receivedBody["top_k"] != float64(10) {
			t.Errorf("expected top_k 10, got %v", receivedBody["top_k"])
		}
		if receivedBody["max_tokens"] != float64(1000) {
			t.Errorf("expected max_tokens 1000, got %v", receivedBody["max_tokens"])
		}
		if receivedBody["stream"] != true {
			t.Errorf("expected stream true, got %v", receivedBody["stream"])
		}
	})
}

func TestAnthropicModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","usage":{"input_tokens":4,"output_tokens":0}}}`,
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

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		// Check API key header
		if receivedHeaders.Get("x-api-key") != "test-api-key" {
			t.Errorf("expected x-api-key header, got %s", receivedHeaders.Get("x-api-key"))
		}

		// Check anthropic-version header
		if receivedHeaders.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("expected anthropic-version header, got %s", receivedHeaders.Get("anthropic-version"))
		}

		// Check custom provider header
		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		// Check custom request header
		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestAnthropicModel_ErrorResponse(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"Invalid API key"}}`))
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error creating stream: %v", err)
		}

		var errorEvent *stream.ErrorEvent
		for event := range events {
			if event.Type == stream.EventError {
				if e, ok := event.Data.(stream.ErrorEvent); ok {
					errorEvent = &e
				}
			}
		}

		if errorEvent == nil {
			t.Fatal("expected error event")
		}

		if errorEvent.Error == nil {
			t.Fatal("expected error in error event")
		}
	})
}

func TestAnthropicModel_ToolCalls(t *testing.T) {
	t.Run("should extract tool calls", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","usage":{"input_tokens":10,"output_tokens":0}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"San Francisco\"}"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`,
				`{"type":"message_stop"}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("What's the weather in San Francisco?"),
			},
			Tools: []tool.Tool{
				{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string"}}}`),
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var toolCallEvent *stream.ToolCallEvent
		var finishReason stream.FinishReason
		for event := range events {
			if event.Type == stream.EventToolCall {
				if tc, ok := event.Data.(stream.ToolCallEvent); ok {
					toolCallEvent = &tc
				}
			}
			if event.Type == stream.EventFinish {
				if f, ok := event.Data.(stream.FinishEvent); ok {
					finishReason = f.FinishReason
				}
			}
		}

		if toolCallEvent == nil {
			t.Fatal("expected tool call event")
		}

		if toolCallEvent.ToolName != "get_weather" {
			t.Errorf("expected tool name get_weather, got %s", toolCallEvent.ToolName)
		}

		if toolCallEvent.ToolCallID != "toolu_01" {
			t.Errorf("expected tool call ID toolu_01, got %s", toolCallEvent.ToolCallID)
		}

		var input map[string]any
		if err := json.Unmarshal(toolCallEvent.Input, &input); err != nil {
			t.Fatalf("failed to unmarshal input: %v", err)
		}

		if input["location"] != "San Francisco" {
			t.Errorf("expected location San Francisco, got %v", input["location"])
		}

		if finishReason != stream.FinishReasonToolCalls {
			t.Errorf("expected finish reason tool_calls, got %v", finishReason)
		}
	})
}

func TestAnthropicModel_FinishReason(t *testing.T) {
	tests := []struct {
		name           string
		stopReason     string
		expectedReason stream.FinishReason
	}{
		{"end_turn should map to stop", "end_turn", stream.FinishReasonStop},
		{"max_tokens should map to length", "max_tokens", stream.FinishReasonLength},
		{"stop_sequence should map to stop", "stop_sequence", stream.FinishReasonStop},
		{"tool_use should map to tool_calls", "tool_use", stream.FinishReasonToolCalls},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)

				chunks := []string{
					`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","usage":{"input_tokens":4,"output_tokens":0}}}`,
					`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
					`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
					`{"type":"content_block_stop","index":0}`,
					`{"type":"message_delta","delta":{"stop_reason":"` + tc.stopReason + `"},"usage":{"output_tokens":1}}`,
					`{"type":"message_stop"}`,
				}

				for _, chunk := range chunks {
					w.Write([]byte("data: " + chunk + "\n\n"))
				}
			}))
			defer server.Close()

			provider := createTestProvider(server.URL)
			model := provider.Model("claude-3-haiku-20240307")

			events, err := model.Stream(context.Background(), &stream.CallOptions{
				Messages: []message.Message{
					message.NewUserMessage("Hello"),
				},
			})

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var finishReason stream.FinishReason
			for event := range events {
				if event.Type == stream.EventFinish {
					if f, ok := event.Data.(stream.FinishEvent); ok {
						finishReason = f.FinishReason
					}
				}
			}

			if finishReason != tc.expectedReason {
				t.Errorf("expected finish reason %v, got %v", tc.expectedReason, finishReason)
			}
		})
	}
}

func TestAnthropicModel_SystemMessage(t *testing.T) {
	t.Run("should pass system message", func(t *testing.T) {
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("You are a helpful assistant."),
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if receivedBody["system"] != "You are a helpful assistant." {
			t.Errorf("expected system message, got %v", receivedBody["system"])
		}
	})
}

// Tests for ProviderOptions

func TestAnthropicModel_Thinking(t *testing.T) {
	t.Run("should send thinking provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-7-sonnet-20250219","usage":{"input_tokens":10,"output_tokens":0}}}`,
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-7-sonnet-20250219")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"thinking": map[string]any{
					"type":         "enabled",
					"budgetTokens": 2048,
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		thinking, ok := receivedBody["thinking"].(map[string]any)
		if !ok {
			t.Fatal("expected thinking object in request")
		}
		if thinking["type"] != "enabled" {
			t.Errorf("expected thinking type 'enabled', got %v", thinking["type"])
		}
		if thinking["budget_tokens"] != float64(2048) {
			t.Errorf("expected budget_tokens 2048, got %v", thinking["budget_tokens"])
		}
	})
}

func TestAnthropicModel_CacheControl(t *testing.T) {
	t.Run("should send cacheControl provider option with system message", func(t *testing.T) {
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				{
					Role:    message.RoleSystem,
					Content: message.Content{Text: "You are a helpful assistant."},
					ProviderOptions: map[string]any{
						"anthropic": map[string]any{
							"cacheControl": map[string]any{
								"type": "ephemeral",
							},
						},
					},
				},
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		// With cacheControl, system should be an array of blocks
		systemArr, ok := receivedBody["system"].([]any)
		if !ok {
			t.Fatal("expected system to be array with cache_control")
		}
		if len(systemArr) != 1 {
			t.Fatalf("expected 1 system block, got %d", len(systemArr))
		}
		block := systemArr[0].(map[string]any)
		if block["type"] != "text" {
			t.Errorf("expected type 'text', got %v", block["type"])
		}
		if block["text"] != "You are a helpful assistant." {
			t.Errorf("expected text content, got %v", block["text"])
		}
		cacheControl := block["cache_control"].(map[string]any)
		if cacheControl["type"] != "ephemeral" {
			t.Errorf("expected cache_control type 'ephemeral', got %v", cacheControl["type"])
		}
	})
}

func TestAnthropicModel_CacheControl_PerPart(t *testing.T) {
	ephemeralOpts := map[string]any{
		"anthropic": map[string]any{
			"cacheControl": map[string]any{"type": "ephemeral"},
		},
	}

	t.Run("user text part with cache control", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "first"},
				message.TextPart{Text: "cached", ProviderOptions: ephemeralOpts},
				message.TextPart{Text: "last"},
			},
		}
		result := convertToAnthropicContent(content, nil)
		if len(result) != 3 {
			t.Fatalf("expected 3 blocks, got %d", len(result))
		}
		if result[0].CacheControl != nil {
			t.Error("first block should not have cache_control")
		}
		if result[1].CacheControl == nil || result[1].CacheControl.Type != "ephemeral" {
			t.Error("second block should have ephemeral cache_control")
		}
		if result[2].CacheControl != nil {
			t.Error("third block should not have cache_control")
		}
	})

	t.Run("message-level cache control applied to last part", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "first"},
				message.TextPart{Text: "last"},
			},
		}
		result := convertToAnthropicContent(content, ephemeralOpts)
		if result[0].CacheControl != nil {
			t.Error("first block should not have cache_control")
		}
		if result[1].CacheControl == nil || result[1].CacheControl.Type != "ephemeral" {
			t.Error("last block should have message-level cache_control")
		}
	})

	t.Run("part-level overrides message-level", func(t *testing.T) {
		partOpts := map[string]any{
			"anthropic": map[string]any{
				"cacheControl": map[string]any{"type": "ephemeral", "ttl": "1h"},
			},
		}
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "only", ProviderOptions: partOpts},
			},
		}
		// Message-level has no TTL, part-level has "1h"
		result := convertToAnthropicContent(content, ephemeralOpts)
		if result[0].CacheControl == nil {
			t.Fatal("block should have cache_control")
		}
		if result[0].CacheControl.TTL != "1h" {
			t.Errorf("expected TTL '1h' from part-level, got %q", result[0].CacheControl.TTL)
		}
	})

	t.Run("user image part with cache control", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.ImagePart{Image: "abc", MimeType: "image/png", ProviderOptions: ephemeralOpts},
			},
		}
		result := convertToAnthropicContent(content, nil)
		if result[0].CacheControl == nil || result[0].CacheControl.Type != "ephemeral" {
			t.Error("image block should have cache_control")
		}
	})

	t.Run("assistant text and tool_use with cache control", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "thinking..."},
				message.ToolCallPart{
					ID:              "tc1",
					Name:            "search",
					Input:           json.RawMessage(`{"q":"test"}`),
					ProviderOptions: ephemeralOpts,
				},
			},
		}
		result := convertAssistantContent(content, nil)
		if len(result) != 2 {
			t.Fatalf("expected 2 blocks, got %d", len(result))
		}
		if result[0].CacheControl != nil {
			t.Error("text block should not have cache_control")
		}
		if result[1].CacheControl == nil || result[1].CacheControl.Type != "ephemeral" {
			t.Error("tool_use block should have cache_control")
		}
	})

	t.Run("assistant message-level cache control on last block", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "text"},
				message.ToolCallPart{ID: "tc1", Name: "search", Input: json.RawMessage(`{}`)},
			},
		}
		result := convertAssistantContent(content, ephemeralOpts)
		if result[0].CacheControl != nil {
			t.Error("first block should not have cache_control")
		}
		if result[1].CacheControl == nil {
			t.Error("last block should have message-level cache_control")
		}
	})

	t.Run("tool result with cache control", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID:      "tc1",
						ToolName:        "search",
						Output:          message.TextOutput{Value: "results here"},
						ProviderOptions: ephemeralOpts,
					},
				},
			},
		}
		result := convertToolMessages(msg)
		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		// The tool_result block is the first (and only) content block in the message.
		block := result[0].Content[0]
		if block.CacheControl == nil || block.CacheControl.Type != "ephemeral" {
			t.Error("tool_result block should have cache_control")
		}
	})

	t.Run("tool result message-level cache control", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "tc1", ToolName: "search", Output: message.TextOutput{Value: "ok"}},
				},
			},
			ProviderOptions: ephemeralOpts,
		}
		result := convertToolMessages(msg)
		block := result[0].Content[0]
		if block.CacheControl == nil {
			t.Error("tool_result should have message-level cache_control")
		}
	})

	t.Run("tool definition with cache control", func(t *testing.T) {
		tools := []tool.Tool{
			{
				Name:            "search",
				Description:     "Search the web",
				InputSchema:     json.RawMessage(`{"type":"object"}`),
				ProviderOptions: ephemeralOpts,
			},
			{
				Name:        "read",
				Description: "Read a file",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			},
		}
		result, _ := convertToAnthropicTools(tools)
		t0 := result[0].(anthropicTool)
		t1 := result[1].(anthropicTool)
		if t0.CacheControl == nil || t0.CacheControl.Type != "ephemeral" {
			t.Error("first tool should have cache_control")
		}
		if t1.CacheControl != nil {
			t.Error("second tool should not have cache_control")
		}
	})

	t.Run("cache_control key variant accepted", func(t *testing.T) {
		// Test snake_case variant
		snakeOpts := map[string]any{
			"anthropic": map[string]any{
				"cache_control": map[string]any{"type": "ephemeral"},
			},
		}
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "cached", ProviderOptions: snakeOpts},
			},
		}
		result := convertToAnthropicContent(content, nil)
		if result[0].CacheControl == nil || result[0].CacheControl.Type != "ephemeral" {
			t.Error("should accept cache_control (snake_case)")
		}
	})
}

func TestAnthropicModel_MCPServers(t *testing.T) {
	t.Run("should send mcpServers provider option", func(t *testing.T) {
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"mcpServers": []map[string]any{
					{
						"type": "url",
						"name": "test-server",
						"url":  "https://mcp.example.com",
					},
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		mcpServers, ok := receivedBody["mcp_servers"].([]any)
		if !ok {
			t.Fatal("expected mcp_servers array in request")
		}
		if len(mcpServers) != 1 {
			t.Fatalf("expected 1 MCP server, got %d", len(mcpServers))
		}
		server1 := mcpServers[0].(map[string]any)
		if server1["type"] != "url" {
			t.Errorf("expected type 'url', got %v", server1["type"])
		}
		if server1["name"] != "test-server" {
			t.Errorf("expected name 'test-server', got %v", server1["name"])
		}
		if server1["url"] != "https://mcp.example.com" {
			t.Errorf("expected url 'https://mcp.example.com', got %v", server1["url"])
		}
	})
}

func TestAnthropicModel_Container(t *testing.T) {
	t.Run("should send container provider option", func(t *testing.T) {
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

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"container": map[string]any{
					"id": "container_123",
					"skills": []map[string]any{
						{
							"type":    "anthropic",
							"skillId": "pdf_processing",
						},
					},
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		container, ok := receivedBody["container"].(map[string]any)
		if !ok {
			t.Fatal("expected container object in request")
		}
		if container["id"] != "container_123" {
			t.Errorf("expected container id 'container_123', got %v", container["id"])
		}
		skills := container["skills"].([]any)
		if len(skills) != 1 {
			t.Fatalf("expected 1 skill, got %d", len(skills))
		}
		skill := skills[0].(map[string]any)
		if skill["type"] != "anthropic" {
			t.Errorf("expected skill type 'anthropic', got %v", skill["type"])
		}
		if skill["skill_id"] != "pdf_processing" {
			t.Errorf("expected skill_id 'pdf_processing', got %v", skill["skill_id"])
		}
	})
}

func TestAnthropicModel_ResponseFormat(t *testing.T) {
	// Runs a single streaming call, captures the request body, and returns it.
	runCapture := func(t *testing.T, chunks []string, callOpts *stream.CallOptions) (map[string]any, []stream.Event) {
		t.Helper()
		var captured map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &captured)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("claude-3-haiku-20240307")
		if callOpts.Messages == nil {
			callOpts.Messages = []message.Message{message.NewUserMessage("hi")}
		}
		events, err := model.Stream(context.Background(), callOpts)
		if err != nil {
			t.Fatal(err)
		}
		var collected []stream.Event
		for ev := range events {
			collected = append(collected, ev)
		}
		return captured, collected
	}

	t.Run("schema injects synthetic json tool and forces tool_choice", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
		// Simulate tool_use with the synthetic name; input JSON flows via input_json_delta.
		chunks := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"json","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"name\":"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Ada\"}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":8}}`,
			`{"type":"message_stop"}`,
		}
		body, events := runCapture(t, chunks, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})

		// Request body: tools[] has the synthetic json tool, tool_choice forces it.
		toolsRaw, _ := body["tools"].([]any)
		var jsonTool map[string]any
		for _, t := range toolsRaw {
			tm, _ := t.(map[string]any)
			if tm["name"] == "json" {
				jsonTool = tm
				break
			}
		}
		if jsonTool == nil {
			t.Fatalf("expected synthetic json tool in request tools, got %v", toolsRaw)
		}
		tc, _ := body["tool_choice"].(map[string]any)
		if tc["type"] != "tool" || tc["name"] != "json" {
			t.Errorf("expected tool_choice to force json tool, got %v", tc)
		}

		// Stream: synthetic input surfaces as TextDeltas, no ToolCallEvent for "json".
		var text strings.Builder
		var toolCalls []stream.ToolCallEvent
		var finish stream.FinishEvent
		for _, ev := range events {
			switch d := ev.Data.(type) {
			case stream.TextDeltaEvent:
				text.WriteString(d.Text)
			case stream.ToolCallEvent:
				toolCalls = append(toolCalls, d)
			case stream.FinishEvent:
				finish = d
			}
		}
		if text.String() != `{"name":"Ada"}` {
			t.Errorf("expected synthetic JSON surfaced as text %q, got %q", `{"name":"Ada"}`, text.String())
		}
		if len(toolCalls) != 0 {
			t.Errorf("expected no ToolCallEvents for synthetic json, got %d", len(toolCalls))
		}
		if finish.FinishReason != stream.FinishReasonStop {
			t.Errorf("expected finish reason Stop (overridden from ToolCalls), got %v", finish.FinishReason)
		}
	})

	t.Run("no schema injects JSON instruction into system prompt", func(t *testing.T) {
		chunks := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
			`{"type":"message_stop"}`,
		}
		body, _ := runCapture(t, chunks, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		system, _ := body["system"].(string)
		if !strings.Contains(system, "JSON") {
			t.Errorf("expected injected JSON instruction in system prompt, got %q", system)
		}
		if _, hasTools := body["tools"]; hasTools {
			t.Errorf("expected no tools in request (no schema), got %v", body["tools"])
		}
	})

	t.Run("forceJSONToolChoice overrides caller ToolChoice", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object"}`)
		body, _ := runCapture(t, []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`,
			`{"type":"message_stop"}`,
		}, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
			ToolChoice:     map[string]string{"type": "any"},
			Tools: []tool.Tool{{
				Name:        "search",
				Description: "Search",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			}},
		})
		tc, _ := body["tool_choice"].(map[string]any)
		if tc["name"] != "json" {
			t.Errorf("expected tool_choice overridden to json, got %v", tc)
		}
	})
}

// Translated from ai-sdk PRs #17978c6 (cacheControl), #05b8ca2 (user_id),
// #0a0d29c (speed fast-mode), #61f1a61 (inference_geo), #e49c34d
// (anthropicBeta header). Exercises the new providerOption surfaces via
// the wire payload + HTTP header.
func TestAnthropicModel_NewProviderOptionsOnWire(t *testing.T) {
	var capturedBody map[string]any
	var capturedBeta string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBeta = r.Header.Get("anthropic-beta")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"type":"message_stop"}` + "\n\n"))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Model("claude-opus-4-6")

	events, err := model.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Text: "hi"}}},
		ProviderOptions: map[string]any{
			"cacheControl": map[string]any{
				"type": "ephemeral",
			},
			"metadata":      map[string]any{"userId": "user-abc"},
			"speed":         "fast",
			"inferenceGeo":  "us",
			"anthropicBeta": []any{"custom-beta-1", "custom-beta-2"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}

	if _, ok := capturedBody["cache_control"]; !ok {
		t.Errorf("expected cache_control on wire, got body: %v", capturedBody)
	}
	meta, _ := capturedBody["metadata"].(map[string]any)
	if meta["user_id"] != "user-abc" {
		t.Errorf("metadata.user_id = %v, want user-abc", meta["user_id"])
	}
	if capturedBody["speed"] != "fast" {
		t.Errorf("speed = %v, want fast", capturedBody["speed"])
	}
	if capturedBody["inference_geo"] != "us" {
		t.Errorf("inference_geo = %v, want us", capturedBody["inference_geo"])
	}
	if !strings.Contains(capturedBeta, "custom-beta-1") || !strings.Contains(capturedBeta, "custom-beta-2") {
		t.Errorf("anthropic-beta header = %q, want to contain custom-beta-1 and custom-beta-2", capturedBeta)
	}
}

// Regression for ai-sdk PRs #b9d105f (cache tokens on stream),
// #2445da4 (outputTokens.text breakdown), #8c2b1e1 (raw usage data).
func TestAnthropicModel_UsageMetadataOnFinish(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":80,"cache_creation_input_tokens":20}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42,"output_tokens_by_type":{"text":40,"reasoning":2},"some_future_field":"forward-me"}}`,
			`{"type":"message_stop"}`,
		}
		for _, c := range chunks {
			w.Write([]byte("data: " + c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Model("claude-sonnet-4-6")
	events, err := model.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var usage stream.Usage
	for ev := range events {
		if ev.Type == stream.EventFinish {
			usage = ev.Data.(stream.FinishEvent).Usage
		}
	}
	// Cache + output-breakdown token fields now live on the v3
	// stream.Usage shape directly rather than in
	// providerMetadata.anthropic. The raw usage object stays available
	// via Usage.Raw so callers can still forward Anthropic-specific
	// fields we don't model yet.
	if got := usage.InputTokens.CacheRead; got == nil || *got != 80 {
		t.Errorf("InputTokens.CacheRead = %v, want *int(80)", got)
	}
	if got := usage.InputTokens.CacheWrite; got == nil || *got != 20 {
		t.Errorf("InputTokens.CacheWrite = %v, want *int(20)", got)
	}
	if got := usage.OutputTokens.Text; got == nil || *got != 40 {
		t.Errorf("OutputTokens.Text = %v, want *int(40)", got)
	}
	if got := usage.OutputTokens.Reasoning; got == nil || *got != 2 {
		t.Errorf("OutputTokens.Reasoning = %v, want *int(2)", got)
	}
	if usage.Raw["some_future_field"] != "forward-me" {
		t.Errorf("Raw.some_future_field = %v, want forward-me", usage.Raw["some_future_field"])
	}
}

// Exercises ai-sdk #b094c07: the compact_20260112 context-management
// policy streams a conversation summary as `{type:"compaction",...}`
// content blocks with `compaction_delta` sub-deltas. The first delta
// frame can arrive with content:null (server signal, no text yet); we
// must not error on it. Subsequent frames carry the summary string
// which surfaces as normal text-delta events.
func TestAnthropicModel_CompactionDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","usage":{"input_tokens":100,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":null}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Summary of conversation."}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}`,
			`{"type":"message_stop"}`,
		}
		for _, c := range chunks {
			w.Write([]byte("data: " + c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Model("claude-sonnet-4-6")
	events, err := model.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotText string
	var errorSeen bool
	for ev := range events {
		switch ev.Type {
		case stream.EventTextDelta:
			gotText += ev.Data.(stream.TextDeltaEvent).Text
		case stream.EventError:
			errorSeen = true
		}
	}
	if errorSeen {
		t.Error("compaction_delta with null content must not produce an error")
	}
	if gotText != "Summary of conversation." {
		t.Errorf("text = %q, want %q", gotText, "Summary of conversation.")
	}
}

// Verify the new 4.x model IDs land in Models().
func TestAnthropicProvider_ModelsContainsLatest(t *testing.T) {
	p := New(Options{APIKey: "test"})
	models := p.Models()
	wanted := []string{"claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"}
	have := map[string]bool{}
	for _, m := range models {
		have[m] = true
	}
	for _, w := range wanted {
		if !have[w] {
			t.Errorf("Models() missing %q (got %v)", w, models)
		}
	}
	// Obsolete IDs should be pruned.
	for _, obsolete := range []string{"claude-3-opus-20240229", "claude-3-sonnet-20240229"} {
		if have[obsolete] {
			t.Errorf("Models() still lists obsolete %q", obsolete)
		}
	}
}
