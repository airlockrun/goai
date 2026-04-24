package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

// Translated from ai-sdk/packages/openai/src/embedding/openai-embedding-model.test.ts

var dummyEmbeddings = [][]float64{
	{0.1, 0.2, 0.3, 0.4, 0.5},
	{0.6, 0.7, 0.8, 0.9, 1.0},
}
var testEmbedValues = []string{"sunny day at the beach", "rainy day in the city"}

func TestOpenAIEmbedding_DoEmbed(t *testing.T) {
	t.Run("should extract embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
					{"object": "embedding", "index": 1, "embedding": dummyEmbeddings[1]},
				},
				"model": "text-embedding-3-large",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("text-embedding-3-large")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 2 {
			t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
		}

		for i, emb := range result.Embeddings {
			for j, v := range emb.Values {
				if v != dummyEmbeddings[i][j] {
					t.Errorf("expected embedding[%d][%d] = %f, got %f", i, j, dummyEmbeddings[i][j], v)
				}
			}
		}
	})

	t.Run("should extract usage", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
				},
				"model": "text-embedding-3-large",
				"usage": map[string]int{"prompt_tokens": 20, "total_tokens": 20},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("text-embedding-3-large")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues[:1],
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Usage.Tokens != 20 {
			t.Errorf("expected usage tokens 20, got %d", result.Usage.Tokens)
		}
	})

	t.Run("should pass the model and the values", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
					{"object": "embedding", "index": 1, "embedding": dummyEmbeddings[1]},
				},
				"model": "text-embedding-3-large",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("text-embedding-3-large")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["model"] != "text-embedding-3-large" {
			t.Errorf("expected model text-embedding-3-large, got %v", receivedBody["model"])
		}

		input, ok := receivedBody["input"].([]any)
		if !ok {
			t.Fatalf("expected input to be an array, got %T", receivedBody["input"])
		}

		if len(input) != 2 {
			t.Errorf("expected 2 inputs, got %d", len(input))
		}

		if input[0] != testEmbedValues[0] {
			t.Errorf("expected first input %s, got %v", testEmbedValues[0], input[0])
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
				},
				"model": "text-embedding-3-large",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:       "test-api-key",
			BaseURL:      server.URL,
			Organization: "test-organization",
			Project:      "test-project",
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		embModel := provider.EmbeddingModel("text-embedding-3-large")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues[:1],
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("OpenAI-Organization") != "test-organization" {
			t.Errorf("expected OpenAI-Organization header, got %s", receivedHeaders.Get("OpenAI-Organization"))
		}

		if receivedHeaders.Get("OpenAI-Project") != "test-project" {
			t.Errorf("expected OpenAI-Project header, got %s", receivedHeaders.Get("OpenAI-Project"))
		}

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}
