package mistral

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/mistral/src/mistral-embedding-model.test.ts

var dummyEmbeddings = [][]float64{
	{0.1, 0.2, 0.3, 0.4, 0.5},
	{0.6, 0.7, 0.8, 0.9, 1.0},
}
var testEmbedValues = []string{"sunny day at the beach", "rainy day in the city"}

func TestMistralProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "mistral" {
		t.Errorf("expected provider ID mistral, got %s", provider.ID())
	}
}

func TestMistralProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasMistralLarge := false
	for _, m := range models {
		if m == "mistral-large-latest" {
			hasMistralLarge = true
		}
	}
	if !hasMistralLarge {
		t.Error("expected mistral-large-latest in models list")
	}
}

func TestMistralEmbedding_DoEmbed(t *testing.T) {
	t.Run("should extract embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
					{"object": "embedding", "index": 1, "embedding": dummyEmbeddings[1]},
				},
				"model": "mistral-embed",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("mistral-embed")

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
				"model": "mistral-embed",
				"usage": map[string]int{"prompt_tokens": 20, "total_tokens": 20},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("mistral-embed")

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
				"model": "mistral-embed",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		embModel := provider.EmbeddingModel("mistral-embed")

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

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestMistralEmbedding_MaxEmbeddingsPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	embModel := provider.EmbeddingModel("mistral-embed")

	if embModel.MaxEmbeddingsPerCall() != 16384 {
		t.Errorf("expected max embeddings 16384, got %d", embModel.MaxEmbeddingsPerCall())
	}
}

func TestMistralEmbedding_Dimensions(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	embModel := provider.EmbeddingModel("mistral-embed")

	if embModel.Dimensions() != 1024 {
		t.Errorf("expected dimensions 1024, got %d", embModel.Dimensions())
	}
}

// Tests for ProviderOptions

func TestMistralRequestModifier_SafePrompt(t *testing.T) {
	providerOptions := map[string]any{
		"safePrompt": true,
	}

	extra, _, err := mistralRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["safe_prompt"] != true {
		t.Errorf("expected safe_prompt true, got %v", extra["safe_prompt"])
	}
}

func TestMistralRequestModifier_ParallelToolCalls(t *testing.T) {
	providerOptions := map[string]any{
		"parallelToolCalls": false,
	}

	extra, _, err := mistralRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["parallel_tool_calls"] != false {
		t.Errorf("expected parallel_tool_calls false, got %v", extra["parallel_tool_calls"])
	}
}

func TestMistralRequestModifier_AllOptions(t *testing.T) {
	providerOptions := map[string]any{
		"safePrompt":        true,
		"parallelToolCalls": false,
	}

	extra, _, err := mistralRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["safe_prompt"] != true {
		t.Errorf("expected safe_prompt true, got %v", extra["safe_prompt"])
	}
	if extra["parallel_tool_calls"] != false {
		t.Errorf("expected parallel_tool_calls false, got %v", extra["parallel_tool_calls"])
	}
}

// Regression for ai-sdk #297e685: reasoningEffort providerOption maps to
// `reasoning_effort` on the wire. Enables reasoning on
// mistral-small-latest and similar models.
func TestMistralRequestModifier_ReasoningEffort(t *testing.T) {
	extra, _, err := mistralRequestModifier(map[string]any{
		"reasoningEffort": "high",
	})
	if err != nil {
		t.Fatal(err)
	}
	if extra["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", extra["reasoning_effort"])
	}
}
