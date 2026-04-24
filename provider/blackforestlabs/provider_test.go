package blackforestlabs

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/black-forest-labs/src/black-forest-labs-image-model.test.ts

func TestBlackForestLabsProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "black-forest-labs" {
		t.Errorf("expected provider ID black-forest-labs, got %s", provider.ID())
	}
}

func TestBlackForestLabsProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasFlux := false
	for _, m := range models {
		if strings.Contains(m, "flux") {
			hasFlux = true
		}
	}
	if !hasFlux {
		t.Error("expected 'flux' model in models list")
	}
}

func TestFluxImageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.ImageModel("flux-pro-1.1")

	if model.ID() != "flux-pro-1.1" {
		t.Errorf("expected model ID flux-pro-1.1, got %s", model.ID())
	}
}

func TestFluxImageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.ImageModel("flux-pro-1.1")

	if model.Provider() != "black-forest-labs" {
		t.Errorf("expected provider black-forest-labs, got %s", model.Provider())
	}
}

func TestFluxImageModel_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.ImageModel("flux-pro-1.1")

	if model.MaxImagesPerCall() != 1 {
		t.Errorf("expected max images 1, got %d", model.MaxImagesPerCall())
	}
}

func TestFluxImageModel_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// POST to /flux-pro-1.1 to create task
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/flux-pro-1.1") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id": "req-123",
				})
				return
			}

			// GET to /get_result to poll status
			if r.Method == "GET" && strings.Contains(r.URL.Path, "/get_result") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "Ready",
					"result": map[string]any{
						"sample": serverURL + "/image.png",
					},
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
		imageModel := provider.ImageModel("flux-pro-1.1")

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
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/flux-pro-1.1") {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id": "req-123",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/get_result") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "Ready",
					"result": map[string]any{
						"sample": serverURL + "/image.png",
					},
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
		imageModel := provider.ImageModel("flux-pro-1.1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("X-Key") != "test-api-key" {
			t.Errorf("expected X-Key header 'test-api-key', got %s", receivedHeaders.Get("X-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass aspect ratio option", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/flux-pro-1.1") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id": "req-123",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/get_result") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "Ready",
					"result": map[string]any{
						"sample": serverURL + "/image.png",
					},
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
		imageModel := provider.ImageModel("flux-pro-1.1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      "test",
			AspectRatio: "16:9",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Check that dimensions are set for 16:9
		width, _ := receivedBody["width"].(float64)
		height, _ := receivedBody["height"].(float64)

		// 16:9 aspect ratio should have width > height
		if width == 0 || height == 0 {
			t.Error("expected width and height in request body")
		}
	})
}

func TestFluxImageModel_ErrorHandling(t *testing.T) {
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
		imageModel := provider.ImageModel("flux-pro-1.1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
