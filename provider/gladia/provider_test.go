package gladia

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

// Translated from ai-sdk/packages/gladia/src/gladia-transcription-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func TestGladiaProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "gladia" {
		t.Errorf("expected provider ID gladia, got %s", provider.ID())
	}
}

func TestGladiaProvider_Models(t *testing.T) {
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

func TestGladiaTranscription_Transcribe(t *testing.T) {
	t.Run("should transcribe audio from URL", func(t *testing.T) {
		var serverURL string
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/transcription" && r.Method == "POST" {
				var reqBody map[string]any
				json.NewDecoder(r.Body).Decode(&reqBody)

				if reqBody["audio_url"] != "https://example.com/audio.mp3" {
					t.Errorf("expected audio_url to be set, got %v", reqBody["audio_url"])
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":         "test-id",
					"result_url": serverURL + "/result/test-id",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/result/") && r.Method == "GET" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "done",
					"result": map[string]any{
						"transcription": map[string]any{
							"full_transcript": "Hello, world!",
							"languages":       []map[string]any{{"language": "en"}},
							"utterances": []map[string]any{
								{
									"text":       "Hello, world!",
									"start":      0.0,
									"end":        2.0,
									"confidence": 0.95,
									"words": []map[string]any{
										{"word": "Hello,", "start": 0.0, "end": 0.5, "confidence": 0.97},
										{"word": "world!", "start": 0.6, "end": 1.0, "confidence": 0.99},
									},
								},
							},
						},
						"metadata": map[string]any{
							"audio_duration": 2.0,
						},
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
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/upload" {
				receivedHeaders = r.Header
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]string{
					"audio_url": "https://storage.gladia.io/mock-url",
				})
				return
			}

			if r.URL.Path == "/transcription" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":         "test-id",
					"result_url": serverURL + "/result/test-id",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/result/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "done",
					"result": map[string]any{
						"transcription": map[string]any{
							"full_transcript": "Hello",
							"utterances":      []any{},
						},
						"metadata": map[string]any{"audio_duration": 1.0},
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
		transcriptionModel := provider.TranscriptionModel("default")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testAudioData,
			MimeType: "audio/wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("x-gladia-key") != "test-api-key" {
			t.Errorf("expected x-gladia-key header, got %s", receivedHeaders.Get("x-gladia-key"))
		}
	})

	t.Run("should upload and transcribe audio data", func(t *testing.T) {
		uploadCalled := false
		transcriptCalled := false
		var serverURL string

		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/upload" && r.Method == "POST" {
				uploadCalled = true
				body, _ := io.ReadAll(r.Body)
				if len(body) != len(testAudioData) {
					t.Errorf("expected %d bytes, got %d", len(testAudioData), len(body))
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]string{
					"audio_url": "https://storage.gladia.io/mock-url",
				})
				return
			}

			if r.URL.Path == "/transcription" && r.Method == "POST" {
				transcriptCalled = true
				var reqBody map[string]any
				json.NewDecoder(r.Body).Decode(&reqBody)

				if reqBody["audio_url"] != "https://storage.gladia.io/mock-url" {
					t.Errorf("expected uploaded audio URL, got %v", reqBody["audio_url"])
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":         "test-id",
					"result_url": serverURL + "/result/test-id",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/result/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "done",
					"result": map[string]any{
						"transcription": map[string]any{
							"full_transcript": "Hello, world!",
							"utterances":      []any{},
						},
						"metadata": map[string]any{"audio_duration": 1.5},
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
		transcriptionModel := provider.TranscriptionModel("default")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testAudioData,
			MimeType: "audio/wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !uploadCalled {
			t.Error("expected upload endpoint to be called")
		}

		if !transcriptCalled {
			t.Error("expected transcription endpoint to be called")
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}
	})

	t.Run("should extract segments with word timestamps", func(t *testing.T) {
		var serverURL string
		server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/transcription" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"id":         "test-id",
					"result_url": serverURL + "/result/test-id",
				})
				return
			}

			if strings.HasPrefix(r.URL.Path, "/result/") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				json.NewEncoder(w).Encode(map[string]any{
					"status": "done",
					"result": map[string]any{
						"transcription": map[string]any{
							"full_transcript": "Hello, world!",
							"languages":       []map[string]any{{"language": "en"}},
							"utterances": []map[string]any{
								{
									"text":       "Hello,",
									"start":      0.0,
									"end":        0.5,
									"confidence": 0.97,
									"words": []map[string]any{
										{"word": "Hello,", "start": 0.0, "end": 0.5, "confidence": 0.97},
									},
								},
								{
									"text":       "world!",
									"start":      0.6,
									"end":        1.0,
									"confidence": 0.99,
									"words": []map[string]any{
										{"word": "world!", "start": 0.6, "end": 1.0, "confidence": 0.99},
									},
								},
							},
						},
						"metadata": map[string]any{"audio_duration": 2.0},
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

		if result.Segments[0].Text != "Hello," {
			t.Errorf("expected first segment 'Hello,', got %s", result.Segments[0].Text)
		}

		if result.Segments[0].Start != 0.0 {
			t.Errorf("expected start 0.0, got %f", result.Segments[0].Start)
		}

		if len(result.Segments[0].Words) != 1 {
			t.Errorf("expected 1 word in first segment, got %d", len(result.Segments[0].Words))
		}
	})
}

func TestGladiaTranscription_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("default")

	if transcriptionModel.ID() != "default" {
		t.Errorf("expected model ID default, got %s", transcriptionModel.ID())
	}
}

func TestGladiaTranscription_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	transcriptionModel := provider.TranscriptionModel("default")

	if transcriptionModel.Provider() != "gladia" {
		t.Errorf("expected provider gladia, got %s", transcriptionModel.Provider())
	}
}

func TestGladiaTranscription_ErrorHandling(t *testing.T) {
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
