package huggingface

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

func TestHuggingFaceProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "huggingface" {
		t.Errorf("expected provider ID huggingface, got %s", provider.ID())
	}
}

func TestHuggingFaceProvider_Models(t *testing.T) {
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
		t.Error("expected 'llama' model in models list")
	}
}

func TestHuggingFaceLanguageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.LanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct")

	if m.ID() != "meta-llama/Meta-Llama-3.1-8B-Instruct" {
		t.Errorf("expected model ID meta-llama/Meta-Llama-3.1-8B-Instruct, got %s", m.ID())
	}
}

func TestHuggingFaceLanguageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.LanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct")

	if m.Provider() != "huggingface" {
		t.Errorf("expected provider huggingface, got %s", m.Provider())
	}
}

func TestHuggingFaceLanguageModel_Stream(t *testing.T) {
	t.Run("should stream text response", func(t *testing.T) {
		var receivedBody map[string]any
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			// Send SSE events in Hugging Face format
			w.Write([]byte("data: {\"token\": {\"text\": \"Hello\"}}\n\n"))
			w.Write([]byte("data: {\"token\": {\"text\": \" world\"}}\n\n"))
			w.Write([]byte("data: {\"generated_text\": \"Hello world\"}\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.LanguageModel("test-model")

		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
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

		if len(textParts) == 0 {
			t.Error("expected at least one text delta")
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}
	})

	t.Run("should pass custom headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("data: {\"token\": {\"text\": \"test\"}}\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.LanguageModel("test-model")

		events, _ := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "test"}},
			},
			Headers: map[string]string{
				"Custom-Header": "custom-value",
			},
		})

		for range events {
		}

		if receivedHeaders.Get("Custom-Header") != "custom-value" {
			t.Errorf("expected Custom-Header 'custom-value', got %s", receivedHeaders.Get("Custom-Header"))
		}
	})
}

func TestHuggingFaceLanguageModel_ErrorHandling(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"Invalid API key"}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		m := provider.LanguageModel("test-model")

		events, _ := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "test"}},
			},
		})

		var gotError bool
		for event := range events {
			if event.Type == stream.EventError {
				gotError = true
			}
		}

		if !gotError {
			t.Error("expected error event")
		}
	})
}

func TestHuggingFaceEmbeddingModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

	if m.ID() != "sentence-transformers/all-MiniLM-L6-v2" {
		t.Errorf("expected model ID sentence-transformers/all-MiniLM-L6-v2, got %s", m.ID())
	}
}

func TestHuggingFaceEmbeddingModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

	if m.Provider() != "huggingface" {
		t.Errorf("expected provider huggingface, got %s", m.Provider())
	}
}

func TestHuggingFaceEmbeddingModel_MaxEmbeddingsPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

	if m.MaxEmbeddingsPerCall() != 100 {
		t.Errorf("expected max embeddings 100, got %d", m.MaxEmbeddingsPerCall())
	}
}

func TestHuggingFaceEmbeddingModel_Embed(t *testing.T) {
	t.Run("should embed text", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode([][]float64{
				{0.1, 0.2, 0.3},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.EmbeddingModel("test-model")

		result, err := m.Embed(context.Background(), model.EmbedCallOptions{
			Values: []string{"Hello world"},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 1 {
			t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
		}

		if len(result.Embeddings[0].Values) != 3 {
			t.Errorf("expected 3 dimensions, got %d", len(result.Embeddings[0].Values))
		}
	})

	t.Run("should handle multiple inputs", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode([][]float64{
				{0.1, 0.2, 0.3},
				{0.4, 0.5, 0.6},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.EmbeddingModel("test-model")

		result, err := m.Embed(context.Background(), model.EmbedCallOptions{
			Values: []string{"Hello", "World"},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 2 {
			t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
		}
	})
}

func TestHuggingFaceImageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("stabilityai/stable-diffusion-xl-base-1.0")

	if m.ID() != "stabilityai/stable-diffusion-xl-base-1.0" {
		t.Errorf("expected model ID stabilityai/stable-diffusion-xl-base-1.0, got %s", m.ID())
	}
}

func TestHuggingFaceImageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("stabilityai/stable-diffusion-xl-base-1.0")

	if m.Provider() != "huggingface" {
		t.Errorf("expected provider huggingface, got %s", m.Provider())
	}
}

func TestHuggingFaceImageModel_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("stabilityai/stable-diffusion-xl-base-1.0")

	if m.MaxImagesPerCall() != 1 {
		t.Errorf("expected max images 1, got %d", m.MaxImagesPerCall())
	}
}

func TestHuggingFaceImageModel_Generate(t *testing.T) {
	t.Run("should generate image", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "image/png")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("fake-image-data"))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.ImageModel("test-model")

		result, err := m.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "A beautiful sunset",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}

		if receivedBody["inputs"] != "A beautiful sunset" {
			t.Errorf("expected inputs 'A beautiful sunset', got %v", receivedBody["inputs"])
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "image/png")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("fake-image-data"))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := provider.ImageModel("test-model")

		_, err := m.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Header": "custom-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Header") != "custom-value" {
			t.Errorf("expected Custom-Header, got %s", receivedHeaders.Get("Custom-Header"))
		}
	})
}

func TestHuggingFaceImageModel_ErrorHandling(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"Invalid API key"}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		m := provider.ImageModel("test-model")

		_, err := m.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestHuggingFaceLanguageModel_ResponseFormatInjectsInstruction(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.LanguageModel("test-model")
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:       []message.Message{message.NewUserMessage("hi")},
		ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	inputs, _ := capturedBody["inputs"].(string)
	if !strings.Contains(inputs, "JSON schema:") {
		t.Errorf("expected injected JSON instruction in inputs, got %q", inputs)
	}
}
