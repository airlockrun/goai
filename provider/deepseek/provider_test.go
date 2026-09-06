package deepseek

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk patterns for OpenAI-compatible providers

func TestV4SamplingNormalization(t *testing.T) {
	for _, tc := range []struct {
		name, reasoning, thinking, wantEffort string
		sampling                              bool
	}{
		{"default", "", "", "", false},
		{"neutral reasoning", "xhigh", "", "max", false},
		{"disabled", "", "disabled", "", true},
		{"neutral none", "none", "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, hasTemperature := body["temperature"]
				_, hasTopP := body["top_p"]
				if hasTemperature != tc.sampling || hasTopP != tc.sampling {
					t.Errorf("sampling = %v", body)
				}
				effort, _ := body["reasoning_effort"].(string)
				if effort != tc.wantEffort {
					t.Errorf("effort = %q", effort)
				}
				w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			}))
			defer server.Close()
			temperature, topP := 0.5, 0.9
			opts := &stream.CallOptions{Temperature: &temperature, TopP: &topP, Reasoning: tc.reasoning}
			if tc.thinking != "" {
				opts.ProviderOptions = map[string]any{"thinking": map[string]any{"type": tc.thinking}}
			}
			events, err := New(Options{BaseURL: server.URL}).Model("deepseek-v4-flash").Stream(context.Background(), opts)
			if err != nil {
				t.Fatal(err)
			}
			for event := range events {
				if event.Type == stream.EventError {
					t.Fatal(event.Data)
				}
			}
		})
	}
}

func TestDeepSeekProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "deepseek" {
		t.Errorf("expected provider ID deepseek, got %s", provider.ID())
	}
}

func TestDeepSeekModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":12,"total_tokens":20}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		model := provider.Model("deepseek-chat")

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
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		model := provider.Model("deepseek-chat")

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

		if usage.InputTotal() != 10 {
			t.Errorf("expected prompt tokens 10, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 20 {
			t.Errorf("expected completion tokens 20, got %d", usage.OutputTotal())
		}
	})
}

func TestDeepSeekModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
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
		model := provider.Model("deepseek-chat")

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

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestDeepSeekModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		model := provider.Model("deepseek-chat")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if receivedBody["model"] != "deepseek-chat" {
			t.Errorf("expected model deepseek-chat, got %v", receivedBody["model"])
		}

		messages, ok := receivedBody["messages"].([]any)
		if !ok || len(messages) == 0 {
			t.Error("expected messages in request body")
		}
	})
}

func TestDeepSeekModel_ErrorResponse(t *testing.T) {
	t.Run("should emit error event on API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		model := provider.Model("deepseek-chat")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			return // Error returned directly - test passes
		}

		var gotError bool
		for event := range events {
			if event.Type == stream.EventError {
				gotError = true
				break
			}
		}

		if !gotError {
			t.Error("expected error event in stream")
		}
	})
}

// Tests for ProviderOptions - verifies deepseekRequestModifier wires up options correctly

func TestDeepSeekRequestModifier_ThinkingEnabled(t *testing.T) {
	providerOptions := map[string]any{
		"thinking": map[string]any{
			"type": "enabled",
		},
	}

	extra, _, err := deepseekRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	thinking, ok := extra["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" {
		t.Errorf("expected thinking enabled, got %v", extra["thinking"])
	}
}

func TestDeepSeekRequestModifier_ThinkingDisabled(t *testing.T) {
	providerOptions := map[string]any{
		"thinking": map[string]any{
			"type": "disabled",
		},
	}

	extra, _, err := deepseekRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	thinking, ok := extra["thinking"].(map[string]any)
	if !ok || thinking["type"] != "disabled" {
		t.Errorf("expected thinking disabled, got %v", extra["thinking"])
	}
}

func TestDeepSeekRequestModifier_ThinkingAdaptive(t *testing.T) {
	extra, _, err := deepseekRequestModifier(map[string]any{
		"thinking": map[string]any{"type": "adaptive"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	thinking, ok := extra["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" {
		t.Errorf("expected thinking enabled, got %v", extra["thinking"])
	}
}

func TestDeepSeekRequestModifier_RejectsInvalidThinking(t *testing.T) {
	_, _, err := deepseekRequestModifier(map[string]any{
		"thinking": map[string]any{"type": "sometimes"},
	})
	if err == nil {
		t.Fatal("expected invalid thinking type error")
	}
}

func TestDeepSeekRequestModifier_NoThinking(t *testing.T) {
	providerOptions := map[string]any{}

	extra, _, err := deepseekRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// When no thinking config is provided, rely on the provider default.
	if _, exists := extra["thinking"]; exists {
		t.Error("expected thinking to not be set when config is nil")
	}
}

// Mirrors ai-sdk #14743 / #15235: reasoningEffort passes through to
// reasoning_effort and is dropped when thinking is disabled.
func TestDeepSeekRequestModifier_ReasoningEffort(t *testing.T) {
	tests := []struct {
		name      string
		options   map[string]any
		wantValue any
		wantSet   bool
	}{
		{
			name:      "passes effort through",
			options:   map[string]any{"reasoningEffort": "max"},
			wantValue: "max",
			wantSet:   true,
		},
		{
			name: "passes effort with thinking enabled",
			options: map[string]any{
				"reasoningEffort": "xhigh",
				"thinking":        map[string]any{"type": "enabled"},
			},
			wantValue: "max",
			wantSet:   true,
		},
		{
			name: "suppresses effort when thinking disabled",
			options: map[string]any{
				"reasoningEffort": "high",
				"thinking":        map[string]any{"type": "disabled"},
			},
			wantSet: false,
		},
		{
			name:    "no effort set",
			options: map[string]any{},
			wantSet: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extra, _, err := deepseekRequestModifier(tt.options)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got, ok := extra["reasoning_effort"]
			if ok != tt.wantSet {
				t.Fatalf("reasoning_effort set = %v, want %v", ok, tt.wantSet)
			}
			if tt.wantSet && got != tt.wantValue {
				t.Errorf("reasoning_effort = %v, want %v", got, tt.wantValue)
			}
		})
	}
}
