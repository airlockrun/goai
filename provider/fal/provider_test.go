package fal

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/fal/src/fal-image-model.test.ts

func TestFalProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "fal" {
		t.Errorf("expected provider ID fal, got %s", provider.ID())
	}
}

func TestFalProvider_Models(t *testing.T) {
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
		t.Error("expected flux model in models list")
	}
}

func TestFalImage_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		requestID := "test-request-id"

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" && !strings.Contains(r.URL.Path, "/status") {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusAccepted)
				json.NewEncoder(w).Encode(map[string]any{
					"request_id": requestID,
					"status":     "IN_QUEUE",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/status") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "COMPLETED",
					"response": map[string]any{
						"images": []map[string]any{
							{
								"url":          "https://example.com/image.png",
								"content_type": "image/png",
							},
						},
					},
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

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

		if result.Images[0].URL != "https://example.com/image.png" {
			t.Errorf("expected image URL, got %s", result.Images[0].URL)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusAccepted)
				json.NewEncoder(w).Encode(map[string]any{
					"request_id": "test-id",
					"status":     "IN_QUEUE",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/status") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "COMPLETED",
					"response": map[string]any{
						"images": []map[string]any{
							{"url": "https://example.com/image.png"},
						},
					},
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Key test-api-key" {
			t.Errorf("expected Authorization header 'Key test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass size option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusAccepted)
				json.NewEncoder(w).Encode(map[string]any{
					"request_id": "test-id",
					"status":     "IN_QUEUE",
				})
				return
			}

			if r.Method == "GET" && strings.Contains(r.URL.Path, "/status") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "COMPLETED",
					"response": map[string]any{
						"images": []map[string]any{
							{"url": "https://example.com/image.png"},
						},
					},
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Size:   "1024x768",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		imageSize, ok := receivedBody["image_size"].(map[string]any)
		if !ok {
			t.Error("expected image_size object in request body")
			return
		}

		if imageSize["width"] != float64(1024) {
			t.Errorf("expected width 1024, got %v", imageSize["width"])
		}

		if imageSize["height"] != float64(768) {
			t.Errorf("expected height 768, got %v", imageSize["height"])
		}
	})

	t.Run("should pass aspect ratio option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusAccepted)
				json.NewEncoder(w).Encode(map[string]any{
					"request_id": "test-id",
					"status":     "IN_QUEUE",
				})
				return
			}

			if r.Method == "GET" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "COMPLETED",
					"response": map[string]any{
						"images": []map[string]any{
							{"url": "https://example.com/image.png"},
						},
					},
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      "test",
			AspectRatio: "16:9",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["image_size"] != "16:9" {
			t.Errorf("expected image_size '16:9', got %v", receivedBody["image_size"])
		}
	})

	t.Run("should pass num_images option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				json.NewDecoder(r.Body).Decode(&receivedBody)

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusAccepted)
				json.NewEncoder(w).Encode(map[string]any{
					"request_id": "test-id",
					"status":     "IN_QUEUE",
				})
				return
			}

			if r.Method == "GET" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "COMPLETED",
					"response": map[string]any{
						"images": []map[string]any{
							{"url": "https://example.com/image1.png"},
							{"url": "https://example.com/image2.png"},
						},
					},
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

		result, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			N:      2,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["num_images"] != float64(2) {
			t.Errorf("expected num_images 2, got %v", receivedBody["num_images"])
		}

		if len(result.Images) != 2 {
			t.Errorf("expected 2 images, got %d", len(result.Images))
		}
	})
}

func TestFalImage_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	imageModel := provider.ImageModel("fal-ai/flux/dev")

	if imageModel.ID() != "fal-ai/flux/dev" {
		t.Errorf("expected model ID fal-ai/flux/dev, got %s", imageModel.ID())
	}
}

func TestFalImage_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	imageModel := provider.ImageModel("fal-ai/flux/dev")

	if imageModel.Provider() != "fal" {
		t.Errorf("expected provider fal, got %s", imageModel.Provider())
	}
}

func TestFalImage_MaxImagesPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	imageModel := provider.ImageModel("fal-ai/flux/dev")

	if imageModel.MaxImagesPerCall() != 4 {
		t.Errorf("expected max images 4, got %d", imageModel.MaxImagesPerCall())
	}
}

func TestFalImage_ErrorHandling(t *testing.T) {
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
		imageModel := provider.ImageModel("fal-ai/flux/dev")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}

		if !strings.Contains(err.Error(), "fal API error") {
			t.Errorf("expected fal API error, got %v", err)
		}
	})
}
