package revai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/revai/src/revai-transcription-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func TestRevAIProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "revai" {
		t.Errorf("expected provider ID revai, got %s", provider.ID())
	}
}

func TestRevAIProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasDefault := false
	for _, m := range models {
		if m == "default" {
			hasDefault = true
		}
	}
	if !hasDefault {
		t.Error("expected 'default' model in models list")
	}
}

func TestRevAITranscription_Transcribe(t *testing.T) {
	t.Run("should transcribe audio from URL", func(t *testing.T) {
		jobID := "test-job-id"

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/jobs" && r.Method == "POST" {
				var reqBody map[string]any
				json.NewDecoder(r.Body).Decode(&reqBody)

				if reqBody["media_url"] != "https://example.com/audio.mp3" {
					t.Errorf("expected media_url to be set, got %v", reqBody["media_url"])
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id": jobID,
				})
				return
			}

			if r.URL.Path == "/jobs/"+jobID && r.Method == "GET" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":               jobID,
					"status":           "transcribed",
					"duration_seconds": 2.5,
					"language":         "en",
				})
				return
			}

			if r.URL.Path == "/jobs/"+jobID+"/transcript" && r.Method == "GET" {
				w.Header().Set("Content-Type", "application/vnd.rev.transcript.v1.0+json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"monologues": []map[string]any{
						{
							"elements": []map[string]any{
								{"type": "text", "value": "Hello", "ts": 0.0, "end_ts": 0.5, "confidence": 0.97},
								{"type": "punct", "value": ","},
								{"type": "text", "value": " world", "ts": 0.6, "end_ts": 1.0, "confidence": 0.99},
								{"type": "punct", "value": "!"},
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
		transcriptionModel := provider.TranscriptionModel("default")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}

		if result.Language != "en" {
			t.Errorf("expected language 'en', got %s", result.Language)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/jobs" && r.Method == "POST" {
				receivedHeaders = r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id": "test-id",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/jobs/") && r.Method == "GET" {
				if strings.HasSuffix(r.URL.Path, "/transcript") {
					w.Header().Set("Content-Type", "application/vnd.rev.transcript.v1.0+json")
					w.WriteHeader(http.StatusOK)
					json.NewEncoder(w).Encode(map[string]any{
						"monologues": []any{},
					})
					return
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":               "test-id",
					"status":           "transcribed",
					"duration_seconds": 1.0,
					"language":         "en",
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
		transcriptionModel := provider.TranscriptionModel("default")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
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

	t.Run("should extract segments with timestamps", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/jobs" && r.Method == "POST" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{"id": "test-id"})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/jobs/") && r.Method == "GET" {
				if strings.HasSuffix(r.URL.Path, "/transcript") {
					w.Header().Set("Content-Type", "application/vnd.rev.transcript.v1.0+json")
					w.WriteHeader(http.StatusOK)
					json.NewEncoder(w).Encode(map[string]any{
						"monologues": []map[string]any{
							{
								"elements": []map[string]any{
									{"type": "text", "value": "Hello", "ts": 0.25, "end_ts": 0.65, "confidence": 0.97},
									{"type": "punct", "value": ","},
									{"type": "text", "value": " world", "ts": 0.73, "end_ts": 1.02, "confidence": 0.99},
									{"type": "punct", "value": "!"},
								},
							},
						},
					})
					return
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":               "test-id",
					"status":           "transcribed",
					"duration_seconds": 2.0,
					"language":         "en",
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
		transcriptionModel := provider.TranscriptionModel("default")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Segments) != 2 {
			t.Errorf("expected 2 segments, got %d", len(result.Segments))
			return
		}

		if result.Segments[0].Text != "Hello" {
			t.Errorf("expected first segment 'Hello', got %s", result.Segments[0].Text)
		}

		if result.Segments[0].Start != 0.25 {
			t.Errorf("expected start 0.25, got %f", result.Segments[0].Start)
		}
	})
}

func TestRevAITranscription_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("default")

	if transcriptionModel.ID() != "default" {
		t.Errorf("expected model ID default, got %s", transcriptionModel.ID())
	}
}

func TestRevAITranscription_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("default")

	if transcriptionModel.Provider() != "revai" {
		t.Errorf("expected provider revai, got %s", transcriptionModel.Provider())
	}
}

func TestRevAITranscription_ErrorHandling(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"Unauthorized"}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("default")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
