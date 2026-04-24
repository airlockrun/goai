package luma

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/luma/src/luma-image-model.test.ts

func TestLumaProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "luma" {
		t.Errorf("expected provider ID luma, got %s", provider.ID())
	}
}

func TestLumaProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasPhoton := false
	for _, m := range models {
		if strings.Contains(m, "photon") {
			hasPhoton = true
		}
	}
	if !hasPhoton {
		t.Error("expected 'photon' model in models list")
	}
}

func TestLumaImageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("photon-1")

	if m.ID() != "photon-1" {
		t.Errorf("expected model ID photon-1, got %s", m.ID())
	}
}

func TestLumaImageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("photon-1")

	if m.Provider() != "luma" {
		t.Errorf("expected provider luma, got %s", m.Provider())
	}
}

func TestLumaImageModel_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("photon-1")

	if m.MaxImagesPerCall() != 1 {
		t.Errorf("expected max images 1, got %d", m.MaxImagesPerCall())
	}
}

func TestLumaImageModel_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// POST to /generations/image to create task
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generations/image") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "queued",
				})
				return
			}

			// GET to /generations/{id} to poll status
			if r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/generations/") && !strings.Contains(r.URL.Path, "/image") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "completed",
					"assets": map[string]any{
						"image": serverURL + "/image.png",
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
		imageModel := provider.ImageModel("photon-1")

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
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generations/image") {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "queued",
				})
				return
			}

			// GET to /generations/{id} to poll status
			if r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/generations/") && !strings.Contains(r.URL.Path, "/image") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "completed",
					"assets": map[string]any{
						"image": serverURL + "/image.png",
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
		imageModel := provider.ImageModel("photon-1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass aspect ratio option", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/generations/image") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "queued",
				})
				return
			}

			// GET to /generations/{id} to poll status
			if r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/generations/") && !strings.Contains(r.URL.Path, "/image") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":    "gen-123",
					"state": "completed",
					"assets": map[string]any{
						"image": serverURL + "/image.png",
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
		imageModel := provider.ImageModel("photon-1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      "test",
			AspectRatio: "16:9",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["aspect_ratio"] != "16:9" {
			t.Errorf("expected aspect_ratio '16:9', got %v", receivedBody["aspect_ratio"])
		}
	})
}

func TestLumaImageModel_ErrorHandling(t *testing.T) {
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
		imageModel := provider.ImageModel("photon-1")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
