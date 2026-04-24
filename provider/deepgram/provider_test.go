package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk audio provider test patterns

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8}

func TestDeepgramProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "deepgram" {
		t.Errorf("expected provider ID deepgram, got %s", provider.ID())
	}
}

func TestDeepgramProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasNova2 := false
	for _, m := range models {
		if m == "nova-2" {
			hasNova2 = true
		}
	}
	if !hasNova2 {
		t.Error("expected nova-2 in models list")
	}
}

func TestDeepgramSpeech_Generate(t *testing.T) {
	t.Run("should generate speech", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("aura-asteria-en")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello, world!",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}

		if result.MimeType != "audio/mpeg" {
			t.Errorf("expected mime type audio/mpeg, got %s", result.MimeType)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("aura-asteria-en")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello, world!",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Token test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestDeepgramTranscription_Transcribe(t *testing.T) {
	t.Run("should transcribe audio", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"metadata": map[string]any{
					"request_id": "req-123",
					"duration":   2.5,
				},
				"results": map[string]any{
					"channels": []map[string]any{
						{
							"alternatives": []map[string]any{
								{
									"transcript": "Hello, world!",
									"confidence": 0.99,
									"words": []map[string]any{
										{"word": "Hello,", "start": 0.0, "end": 0.5, "confidence": 0.99},
										{"word": "world!", "start": 0.6, "end": 1.0, "confidence": 0.98},
									},
								},
							},
						},
					},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("nova-2")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testAudioData,
			MimeType: "audio/wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}

		if result.Duration == nil || *result.Duration != 2.5 {
			t.Errorf("expected duration 2.5, got %v", result.Duration)
		}

		if len(result.Segments) != 2 {
			t.Errorf("expected 2 segments, got %d", len(result.Segments))
		}
	})

	t.Run("should pass language", func(t *testing.T) {
		var requestURL string

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestURL = r.URL.String()

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"metadata": map[string]any{
					"request_id": "req-123",
					"duration":   2.0,
				},
				"results": map[string]any{
					"channels": []map[string]any{
						{
							"alternatives": []map[string]any{
								{
									"transcript": "Hola, mundo!",
									"confidence": 0.99,
									"words":      []map[string]any{},
								},
							},
						},
					},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("nova-2")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testAudioData,
			Language: "es",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if requestURL == "" {
			t.Fatal("expected request URL to be captured")
		}

		// URL should contain language=es
		if !contains(requestURL, "language=es") {
			t.Errorf("expected URL to contain language=es, got %s", requestURL)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"metadata": map[string]any{
					"request_id": "req-123",
					"duration":   2.5,
				},
				"results": map[string]any{
					"channels": []map[string]any{
						{
							"alternatives": []map[string]any{
								{
									"transcript": "Hello!",
									"confidence": 0.99,
									"words":      []map[string]any{},
								},
							},
						},
					},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("nova-2")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio: testAudioData,
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Token test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestDeepgramModels_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	speechModel := provider.SpeechModel("aura-asteria-en")
	if speechModel.ID() != "aura-asteria-en" {
		t.Errorf("expected speech model ID aura-asteria-en, got %s", speechModel.ID())
	}

	transcriptionModel := provider.TranscriptionModel("nova-2")
	if transcriptionModel.ID() != "nova-2" {
		t.Errorf("expected transcription model ID nova-2, got %s", transcriptionModel.ID())
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsAt(s, substr, 0))
}

func containsAt(s, substr string, start int) bool {
	if start+len(substr) > len(s) {
		return false
	}
	for i := 0; i < len(substr); i++ {
		if s[start+i] != substr[i] {
			return containsAt(s, substr, start+1)
		}
	}
	return true
}
