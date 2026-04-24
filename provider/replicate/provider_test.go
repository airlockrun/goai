package replicate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

func TestReplicateProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "replicate" {
		t.Errorf("expected provider ID replicate, got %s", provider.ID())
	}
}

func TestReplicateProvider_Models(t *testing.T) {
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

func TestReplicateImageModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("black-forest-labs/flux-1.1-pro")

	if m.ID() != "black-forest-labs/flux-1.1-pro" {
		t.Errorf("expected model ID black-forest-labs/flux-1.1-pro, got %s", m.ID())
	}
}

func TestReplicateImageModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("black-forest-labs/flux-1.1-pro")

	if m.Provider() != "replicate" {
		t.Errorf("expected provider replicate, got %s", m.Provider())
	}
}

func TestReplicateImageModel_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	m := provider.ImageModel("black-forest-labs/flux-1.1-pro")

	if m.MaxImagesPerCall() != 4 {
		t.Errorf("expected max images 4, got %d", m.MaxImagesPerCall())
	}
}

func TestReplicateImageModel_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// POST to /predictions to create prediction
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/predictions") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusCreated)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "starting",
				})
				return
			}

			// GET to /predictions/{id} to poll status
			if r.Method == "GET" && strings.Contains(r.URL.Path, "/predictions/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "succeeded",
					"output": []string{serverURL + "/image.png"},
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
		imageModel := provider.ImageModel("test-version")

		result, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "A beautiful sunset over the ocean",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		input, ok := receivedBody["input"].(map[string]any)
		if !ok {
			t.Fatal("expected input in request body")
		}

		if input["prompt"] != "A beautiful sunset over the ocean" {
			t.Errorf("expected prompt 'A beautiful sunset over the ocean', got %v", input["prompt"])
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/predictions") {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusCreated)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "starting",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/predictions/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "succeeded",
					"output": []string{serverURL + "/image.png"},
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
		imageModel := provider.ImageModel("test-version")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Token test-api-key" {
			t.Errorf("expected Authorization header 'Token test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass aspect ratio option", func(t *testing.T) {
		var receivedBody map[string]any
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/predictions") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusCreated)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "starting",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/predictions/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "succeeded",
					"output": []string{serverURL + "/image.png"},
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
		imageModel := provider.ImageModel("test-version")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      "test",
			AspectRatio: "16:9",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		input, ok := receivedBody["input"].(map[string]any)
		if !ok {
			t.Fatal("expected input in request body")
		}

		if input["aspect_ratio"] != "16:9" {
			t.Errorf("expected aspect_ratio '16:9', got %v", input["aspect_ratio"])
		}
	})

	t.Run("should handle multiple outputs", func(t *testing.T) {
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/predictions") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusCreated)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "starting",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/predictions/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "succeeded",
					"output": []string{
						serverURL + "/image1.png",
						serverURL + "/image2.png",
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
		imageModel := provider.ImageModel("test-version")

		result, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			N:      2,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Images) != 2 {
			t.Errorf("expected 2 images, got %d", len(result.Images))
		}
	})
}

func TestReplicateImageModel_ErrorHandling(t *testing.T) {
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
		imageModel := provider.ImageModel("test-version")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})

	t.Run("should handle prediction failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/predictions") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusCreated)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "starting",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/predictions/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "pred-123",
					"status": "failed",
					"error":  "Model execution failed",
				})
				return
			}

			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imageModel := provider.ImageModel("test-version")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
