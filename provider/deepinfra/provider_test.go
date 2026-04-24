package deepinfra

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/deepinfra/src/deepinfra-provider.test.ts

func TestDeepInfraProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "deepinfra" {
		t.Errorf("expected provider ID deepinfra, got %s", provider.ID())
	}
}

func TestDeepInfraProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasLlama := false
	for _, m := range models {
		if strings.Contains(m, "llama") || strings.Contains(m, "Llama") {
			hasLlama = true
		}
	}
	if !hasLlama {
		t.Error("expected Llama model in models list")
	}
}

func TestDeepInfraModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
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
		langModel := provider.Model("meta-llama/Meta-Llama-3.1-8B-Instruct")

		events, err := langModel.Stream(context.Background(), &stream.CallOptions{
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
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
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
		langModel := provider.Model("meta-llama/Meta-Llama-3.1-8B-Instruct")

		events, err := langModel.Stream(context.Background(), &stream.CallOptions{
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

func TestDeepInfraModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
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
		langModel := provider.Model("meta-llama/Meta-Llama-3.1-8B-Instruct")

		events, err := langModel.Stream(context.Background(), &stream.CallOptions{
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

func TestDeepInfraModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
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
		langModel := provider.Model("meta-llama/Meta-Llama-3.1-8B-Instruct")

		events, err := langModel.Stream(context.Background(), &stream.CallOptions{
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

		if receivedBody["model"] != "meta-llama/Meta-Llama-3.1-8B-Instruct" {
			t.Errorf("expected model meta-llama/Meta-Llama-3.1-8B-Instruct, got %v", receivedBody["model"])
		}

		messages, ok := receivedBody["messages"].([]any)
		if !ok || len(messages) == 0 {
			t.Error("expected messages in request body")
		}
	})
}

func TestDeepInfraEmbedding_Embed(t *testing.T) {
	t.Run("should generate embeddings", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)

			response := map[string]any{
				"data": []map[string]any{
					{
						"embedding": []float64{0.1, 0.2, 0.3, 0.4, 0.5},
						"index":     0,
					},
				},
				"usage": map[string]int{
					"prompt_tokens": 5,
					"total_tokens":  5,
				},
			}
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("BAAI/bge-large-en-v1.5")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: []string{"Hello, world!"},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 1 {
			t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
		}

		if len(result.Embeddings[0].Values) != 5 {
			t.Errorf("expected 5 dimensions, got %d", len(result.Embeddings[0].Values))
		}

		if result.Usage.Tokens != 5 {
			t.Errorf("expected 5 tokens, got %d", result.Usage.Tokens)
		}
	})

	t.Run("should pass headers for embeddings", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)

			response := map[string]any{
				"data": []map[string]any{
					{"embedding": []float64{0.1, 0.2}, "index": 0},
				},
				"usage": map[string]int{"total_tokens": 3},
			}
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("BAAI/bge-large-en-v1.5")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: []string{"test"},
			Headers: map[string]string{
				"Custom-Header": "custom-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Header") != "custom-value" {
			t.Errorf("expected Custom-Header, got %s", receivedHeaders.Get("Custom-Header"))
		}
	})
}

func TestDeepInfraModel_ErrorResponse(t *testing.T) {
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
		langModel := provider.Model("meta-llama/Meta-Llama-3.1-8B-Instruct")

		events, err := langModel.Stream(context.Background(), &stream.CallOptions{
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
