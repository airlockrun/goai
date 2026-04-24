package google

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/google/src/google-generative-ai-embedding-model.test.ts

var dummyEmbeddings = [][]float64{
	{0.1, 0.2, 0.3, 0.4, 0.5},
	{0.6, 0.7, 0.8, 0.9, 1.0},
}
var testEmbedValues = []string{"sunny day at the beach", "rainy day in the city"}

func TestGoogleEmbedding_DoEmbed(t *testing.T) {
	t.Run("should extract embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"embeddings": []map[string]any{
					{"values": dummyEmbeddings[0]},
					{"values": dummyEmbeddings[1]},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("text-embedding-004")

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

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"embeddings": []map[string]any{
					{"values": dummyEmbeddings[0]},
				},
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
		embModel := provider.EmbeddingModel("text-embedding-004")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues[:1],
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Note: Google uses API key in URL, not Authorization header
		// Check custom request header
		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"code":400,"message":"Invalid API key","status":"INVALID_ARGUMENT"}}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		embModel := provider.EmbeddingModel("text-embedding-004")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues,
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestGoogleEmbedding_MaxEmbeddingsPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	embModel := provider.EmbeddingModel("text-embedding-004")

	if embModel.MaxEmbeddingsPerCall() != 100 {
		t.Errorf("expected max embeddings 100, got %d", embModel.MaxEmbeddingsPerCall())
	}
}

func TestGoogleEmbedding_Dimensions(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	tests := []struct {
		modelID    string
		dimensions int
	}{
		{"text-embedding-004", 768},
		{"embedding-001", 768},
		{"unknown-model", 0},
	}

	for _, tc := range tests {
		embModel := provider.EmbeddingModel(tc.modelID)
		if embModel.Dimensions() != tc.dimensions {
			t.Errorf("expected dimensions %d for %s, got %d", tc.dimensions, tc.modelID, embModel.Dimensions())
		}
	}
}
