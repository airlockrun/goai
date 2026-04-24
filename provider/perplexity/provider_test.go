package perplexity

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

// Translated from ai-sdk/packages/perplexity/src/perplexity-language-model.test.ts

func TestPerplexityProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "perplexity" {
		t.Errorf("expected provider ID perplexity, got %s", provider.ID())
	}
}

func TestPerplexityProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}
}

func TestPerplexityModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":", "},"finish_reason":null}]}`,
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":"World!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`,
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
		model := provider.Model("perplexity-001")

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
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`,
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
		model := provider.Model("perplexity-001")

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

func TestPerplexityModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`,
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
		model := provider.Model("perplexity-001")

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

// Note: Citation extraction is not yet implemented in the provider.
// When implemented, tests for extracting citations as sources should be added here.

func TestPerplexityModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"stream-id","created":1680003600,"model":"perplexity-001","choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`,
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
		model := provider.Model("perplexity-001")

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

		if receivedBody["model"] != "perplexity-001" {
			t.Errorf("expected model perplexity-001, got %v", receivedBody["model"])
		}

		messages, ok := receivedBody["messages"].([]any)
		if !ok || len(messages) == 0 {
			t.Error("expected messages in request body")
		}
	})

	// Note: Provider options passthrough is not yet implemented in openaicompat provider.
	// When implemented, this test should verify that perplexity-specific options
	// like search_recency_filter and return_images are passed to the API.
}

func TestPerplexityModel_ErrorResponse(t *testing.T) {
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
		model := provider.Model("perplexity-001")

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
