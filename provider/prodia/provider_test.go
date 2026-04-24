package prodia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/prodia/src/prodia-image-model.test.ts

func TestProdiaProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "prodia" {
		t.Errorf("expected provider ID prodia, got %s", provider.ID())
	}
}

func TestProdiaProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasSDXL := false
	for _, m := range models {
		if strings.Contains(m, "sdxl") || m == "sdxl" {
			hasSDXL = true
		}
	}
	if !hasSDXL {
		t.Error("expected 'sdxl' model in models list")
	}
}

func TestProdiaImageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("sdxl")

	if m.ID() != "sdxl" {
		t.Errorf("expected model ID sdxl, got %s", m.ID())
	}
}

func TestProdiaImageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("sdxl")

	if m.Provider() != "prodia" {
		t.Errorf("expected provider prodia, got %s", m.Provider())
	}
}

func TestProdiaImageModel_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("sdxl")

	if m.MaxImagesPerCall() != 1 {
		t.Errorf("expected max images 1, got %d", m.MaxImagesPerCall())
	}
}

func TestProdiaImageModel_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// POST to /sdxl/generate to create task
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generate") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":    "job-123",
					"status": "queued",
				})
				return
			}

			// GET to /job/{id} to poll status
			if r.Method == "GET" && strings.Contains(r.URL.Path, "/job/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":      "job-123",
					"status":   "succeeded",
					"imageUrl": serverURL + "/image.png",
				})
				return
			}

			w.WriteHeader(http.StatusNotFound)
		}))
		server.Start()
		serverURL = server.URL
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imageModel := provider.ImageModel("sdxl")

		result, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "A beautiful sunset over the ocean",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["prompt"] != "A beautiful sunset over the ocean" {
			t.Errorf("expected prompt 'A beautiful sunset over the ocean', got %v", receivedBody["prompt"])
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generate") {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":    "job-123",
					"status": "queued",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/job/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":      "job-123",
					"status":   "succeeded",
					"imageUrl": serverURL + "/image.png",
				})
				return
			}

			w.WriteHeader(http.StatusNotFound)
		}))
		server.Start()
		serverURL = server.URL
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imageModel := provider.ImageModel("sdxl")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("X-Prodia-Key") != "test-api-key" {
			t.Errorf("expected X-Prodia-Key header 'test-api-key', got %s", receivedHeaders.Get("X-Prodia-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass size option", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generate") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":    "job-123",
					"status": "queued",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/job/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"job":      "job-123",
					"status":   "succeeded",
					"imageUrl": serverURL + "/image.png",
				})
				return
			}

			w.WriteHeader(http.StatusNotFound)
		}))
		server.Start()
		serverURL = server.URL
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imageModel := provider.ImageModel("sdxl")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Size:   "1024x768",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		width, _ := receivedBody["width"].(float64)
		height, _ := receivedBody["height"].(float64)

		if width != 1024 {
			t.Errorf("expected width 1024, got %v", width)
		}

		if height != 768 {
			t.Errorf("expected height 768, got %v", height)
		}
	})
}

func TestProdiaImageModel_ErrorHandling(t *testing.T) {
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
		imageModel := provider.ImageModel("sdxl")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
