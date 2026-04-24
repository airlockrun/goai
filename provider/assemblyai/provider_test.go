package assemblyai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/assemblyai/src/assemblyai-transcription-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func TestAssemblyAIProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "assemblyai" {
		t.Errorf("expected provider ID assemblyai, got %s", provider.ID())
	}
}

func TestAssemblyAIProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasBest := false
	for _, m := range models {
		if m == "best" {
			hasBest = true
		}
	}
	if !hasBest {
		t.Error("expected 'best' model in models list")
	}
}

func TestAssemblyAITranscription_Transcribe(t *testing.T) {
	t.Run("should transcribe audio from URL", func(t *testing.T) {
		transcriptID := "test-transcript-id"
		callCount := 0

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/transcript" && r.Method == "POST" {
				// Create transcript request
				var reqBody map[string]any
				json.NewDecoder(r.Body).Decode(&reqBody)

				if reqBody["audio_url"] != "https://example.com/audio.mp3" {
					t.Errorf("expected audio_url to be set, got %v", reqBody["audio_url"])
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     transcriptID,
					"status": "completed",
					"text":   "Hello, world!",
					"words": []map[string]any{
						{"text": "Hello,", "start": 250, "end": 650, "confidence": 0.97},
						{"text": "world!", "start": 730, "end": 1022, "confidence": 0.99},
					},
					"audio_duration": 2.0,
					"language_code":  "en_us",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/transcript/") && r.Method == "GET" {
				callCount++
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     transcriptID,
					"status": "completed",
					"text":   "Hello, world!",
					"words": []map[string]any{
						{"text": "Hello,", "start": 250, "end": 650, "confidence": 0.97},
						{"text": "world!", "start": 730, "end": 1022, "confidence": 0.99},
					},
					"audio_duration": 2.0,
					"language_code":  "en_us",
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
		transcriptionModel := provider.TranscriptionModel("best")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/upload" {
				receivedHeaders = r.Header
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]string{
					"upload_url": "https://storage.assemblyai.com/mock-url",
				})
				return
			}

			if r.URL.Path == "/transcript" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":             "test-id",
					"status":         "completed",
					"text":           "Hello",
					"words":          []any{},
					"audio_duration": 1.0,
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
		transcriptionModel := provider.TranscriptionModel("best")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio: testAudioData,
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Content-Type") != "application/octet-stream" {
			t.Errorf("expected Content-Type application/octet-stream, got %s", receivedHeaders.Get("Content-Type"))
		}
	})

	t.Run("should upload and transcribe audio data", func(t *testing.T) {
		uploadCalled := false
		transcriptCalled := false

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/upload" && r.Method == "POST" {
				uploadCalled = true
				body, _ := io.ReadAll(r.Body)
				if len(body) != len(testAudioData) {
					t.Errorf("expected %d bytes, got %d", len(testAudioData), len(body))
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]string{
					"upload_url": "https://storage.assemblyai.com/mock-url",
				})
				return
			}

			if r.URL.Path == "/transcript" && r.Method == "POST" {
				transcriptCalled = true
				var reqBody map[string]any
				json.NewDecoder(r.Body).Decode(&reqBody)

				if reqBody["audio_url"] != "https://storage.assemblyai.com/mock-url" {
					t.Errorf("expected uploaded audio URL, got %v", reqBody["audio_url"])
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":             "test-id",
					"status":         "completed",
					"text":           "Hello, world!",
					"words":          []any{},
					"audio_duration": 1.5,
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
		transcriptionModel := provider.TranscriptionModel("best")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio: testAudioData,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !uploadCalled {
			t.Error("expected upload endpoint to be called")
		}

		if !transcriptCalled {
			t.Error("expected transcript endpoint to be called")
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}
	})

	t.Run("should extract segments with timestamps", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/transcript" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":     "test-id",
					"status": "completed",
					"text":   "Hello, world!",
					"words": []map[string]any{
						{"text": "Hello,", "start": 250, "end": 650, "confidence": 0.97},
						{"text": "world!", "start": 730, "end": 1022, "confidence": 0.99},
					},
					"audio_duration": 2.0,
					"language_code":  "en_us",
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
		transcriptionModel := provider.TranscriptionModel("best")

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

		if result.Segments[0].Text != "Hello," {
			t.Errorf("expected first segment 'Hello,', got %s", result.Segments[0].Text)
		}

		// Time is in seconds (250ms = 0.25s)
		if result.Segments[0].Start != 0.25 {
			t.Errorf("expected start 0.25, got %f", result.Segments[0].Start)
		}
	})
}

func TestAssemblyAITranscription_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("best")

	if transcriptionModel.ID() != "best" {
		t.Errorf("expected model ID best, got %s", transcriptionModel.ID())
	}
}

func TestAssemblyAITranscription_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("best")

	if transcriptionModel.Provider() != "assemblyai" {
		t.Errorf("expected provider assemblyai, got %s", transcriptionModel.Provider())
	}
}

func TestAssemblyAITranscription_ErrorHandling(t *testing.T) {
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
		transcriptionModel := provider.TranscriptionModel("best")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			AudioURL: "https://example.com/audio.mp3",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
