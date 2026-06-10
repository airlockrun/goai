package xai

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

// Translated from ai-sdk/packages/xai/src/xai-chat-language-model.test.ts

func TestXaiProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "xai" {
		t.Errorf("expected provider ID xai, got %s", provider.ID())
	}
}

func TestXaiProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasGrok := false
	for _, m := range models {
		if m == "grok-beta" || strings.Contains(m, "grok") {
			hasGrok = true
		}
	}
	if !hasGrok {
		t.Error("expected grok model in models list")
	}
}

func TestXaiModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":32,"total_tokens":36}}`,
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
		model := provider.Model("grok-beta")

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
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
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
		model := provider.Model("grok-beta")

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
		if usage.OutputTotal() != 5 {
			t.Errorf("expected completion tokens 5, got %d", usage.OutputTotal())
		}
	})
}

func TestXaiModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
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
		model := provider.Model("grok-beta")

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

func TestXaiModel_ToolCalls(t *testing.T) {
	t.Run("should extract tool calls", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"a9648117","object":"chat.completion.chunk","created":1750535985,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"a9648117","object":"chat.completion.chunk","created":1750535985,"model":"grok-beta","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"id":"call_yfBEybNYi","type":"function","function":{"name":"test-tool","arguments":"{\"value\":\"Sparkle Day\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":183,"total_tokens":316,"completion_tokens":133}}`,
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
		model := provider.Model("grok-beta")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var toolCalls []stream.ToolCallEvent
		for event := range events {
			if event.Type == stream.EventToolCall {
				if tc, ok := event.Data.(stream.ToolCallEvent); ok {
					toolCalls = append(toolCalls, tc)
				}
			}
		}

		if len(toolCalls) != 1 {
			t.Errorf("expected 1 tool call, got %d", len(toolCalls))
		}

		if toolCalls[0].ToolName != "test-tool" {
			t.Errorf("expected tool name test-tool, got %s", toolCalls[0].ToolName)
		}
	})
}

func TestXaiModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"35e18f56","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
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
		model := provider.Model("grok-beta")

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

		if receivedBody["model"] != "grok-beta" {
			t.Errorf("expected model grok-beta, got %v", receivedBody["model"])
		}

		messages, ok := receivedBody["messages"].([]any)
		if !ok || len(messages) == 0 {
			t.Error("expected messages in request body")
		}
	})
}

func TestXaiModel_FinishReason(t *testing.T) {
	tests := []struct {
		name            string
		apiFinishReason string
		expectedFinish  stream.FinishReason
	}{
		{"stop should map to stop", "stop", stream.FinishReasonStop},
		{"length should map to length", "length", stream.FinishReasonLength},
		{"tool_calls should map to tool_calls", "tool_calls", stream.FinishReasonToolCalls},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)

				chunks := []string{
					`data: {"id":"1","object":"chat.completion.chunk","created":1750537778,"model":"grok-beta","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"` + tc.apiFinishReason + `"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
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
			model := provider.Model("grok-beta")

			events, _ := model.Stream(context.Background(), &stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("Hi")},
			})

			var finishReason stream.FinishReason
			for event := range events {
				if event.Type == stream.EventFinish {
					if finish, ok := event.Data.(stream.FinishEvent); ok {
						finishReason = finish.FinishReason
					}
				}
			}

			if finishReason != tc.expectedFinish {
				t.Errorf("expected finish reason %s, got %s", tc.expectedFinish, finishReason)
			}
		})
	}
}

func TestXaiModel_ErrorResponse(t *testing.T) {
	t.Run("should emit error event on API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		model := provider.Model("grok-beta")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Stream may return nil error but emit error events
		if err != nil {
			return // Error returned directly - test passes
		}

		// Check for error event in the stream
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

// Tests for ProviderOptions - verifies xaiRequestModifier wires up options correctly

func TestXaiRequestModifier_ReasoningEffort(t *testing.T) {
	providerOptions := map[string]any{
		"reasoningEffort": "high",
	}

	extra, _, err := xaiRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["reasoning_effort"] != "high" {
		t.Errorf("expected reasoning_effort 'high', got %v", extra["reasoning_effort"])
	}
}

func TestXaiRequestModifier_ParallelFunctionCalling(t *testing.T) {
	falseVal := false
	providerOptions := map[string]any{
		"parallel_function_calling": falseVal,
	}

	extra, _, err := xaiRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["parallel_function_calling"] != false {
		t.Errorf("expected parallel_function_calling false, got %v", extra["parallel_function_calling"])
	}
}

func TestXaiRequestModifier_SearchParameters(t *testing.T) {
	trueVal := true
	providerOptions := map[string]any{
		"searchParameters": map[string]any{
			"mode":             "on",
			"returnCitations":  trueVal,
			"fromDate":         "2024-01-01",
			"toDate":           "2024-12-31",
			"maxSearchResults": 30,
			"sources": []map[string]any{
				{"type": "web"},
				{"type": "x"},
			},
		},
	}

	extra, _, err := xaiRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	searchParams, ok := extra["search_parameters"].(map[string]any)
	if !ok {
		t.Fatal("expected search_parameters to be a map")
	}

	if searchParams["mode"] != "on" {
		t.Errorf("expected mode 'on', got %v", searchParams["mode"])
	}
	if searchParams["return_citations"] != true {
		t.Errorf("expected return_citations true, got %v", searchParams["return_citations"])
	}
	if searchParams["from_date"] != "2024-01-01" {
		t.Errorf("expected from_date '2024-01-01', got %v", searchParams["from_date"])
	}
	if searchParams["to_date"] != "2024-12-31" {
		t.Errorf("expected to_date '2024-12-31', got %v", searchParams["to_date"])
	}
	if searchParams["max_search_results"] != 30 {
		t.Errorf("expected max_search_results 30, got %v", searchParams["max_search_results"])
	}
}

func TestXaiRequestModifier_AllOptions(t *testing.T) {
	trueVal := true
	providerOptions := map[string]any{
		"reasoningEffort":           "low",
		"parallel_function_calling": trueVal,
		"searchParameters": map[string]any{
			"mode": "auto",
		},
	}

	extra, _, err := xaiRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["reasoning_effort"] != "low" {
		t.Errorf("expected reasoning_effort 'low', got %v", extra["reasoning_effort"])
	}
	if extra["parallel_function_calling"] != true {
		t.Errorf("expected parallel_function_calling true, got %v", extra["parallel_function_calling"])
	}

	searchParams, ok := extra["search_parameters"].(map[string]any)
	if !ok {
		t.Fatal("expected search_parameters to be a map")
	}
	if searchParams["mode"] != "auto" {
		t.Errorf("expected mode 'auto', got %v", searchParams["mode"])
	}
}

// Regression for ai-sdk #2e00e03: logprobs + topLogprobs providerOptions
// forward as logprobs/top_logprobs on the xAI chat request.
func TestXaiRequestModifier_Logprobs(t *testing.T) {
	opts := map[string]any{
		"logprobs":    true,
		"topLogprobs": float64(5), // arrives via JSON round-trip as float64
	}
	extra, _, err := xaiRequestModifier(opts)
	if err != nil {
		t.Fatal(err)
	}
	if extra["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", extra["logprobs"])
	}
	if extra["top_logprobs"] != 5 {
		t.Errorf("top_logprobs = %v, want 5", extra["top_logprobs"])
	}
}

// Verify the curated Grok lineup (ai-sdk #15276): only the current
// autocomplete IDs are advertised, retired variants are gone.
func TestXaiProvider_ModelsContainsLatest(t *testing.T) {
	p := New(Options{APIKey: "k"})
	have := map[string]bool{}
	for _, m := range p.Models() {
		have[m] = true
	}
	for _, w := range []string{
		"grok-4.20-non-reasoning",
		"grok-4.20-reasoning",
		"grok-4.3",
		"grok-latest",
	} {
		if !have[w] {
			t.Errorf("Models() missing %q", w)
		}
	}
	for _, retired := range []string{
		"grok-2", "grok-beta", "grok-3", "grok-3-mini",
		"grok-4", "grok-4-1-fast-reasoning", "grok-code-fast-1",
		"grok-4.20-0309-reasoning", "grok-4.20-multi-agent-0309",
	} {
		if have[retired] {
			t.Errorf("Models() still lists retired %q", retired)
		}
	}
}
