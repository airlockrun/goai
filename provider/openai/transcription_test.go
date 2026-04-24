package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

// Translated from ai-sdk/packages/openai/src/transcription/openai-transcription-model.test.ts

var testTranscriptionAudio = []byte{1, 2, 3, 4, 5, 6, 7, 8}

func TestOpenAITranscription_DoTranscribe(t *testing.T) {
	t.Run("should transcribe audio", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"text":     "Hello, world!",
				"language": "en",
				"duration": 2.5,
				"segments": []map[string]any{
					{
						"id":    0,
						"text":  "Hello, world!",
						"start": 0.0,
						"end":   2.5,
					},
				},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("whisper-1")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testTranscriptionAudio,
			Filename: "test.wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}

		if result.Language != "en" {
			t.Errorf("expected language en, got %s", result.Language)
		}

		if result.Duration == nil || *result.Duration != 2.5 {
			t.Errorf("expected duration 2.5, got %v", result.Duration)
		}

		if len(result.Segments) != 1 {
			t.Errorf("expected 1 segment, got %d", len(result.Segments))
		}
	})

	t.Run("should pass language", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Parse multipart form to check language field
			r.ParseMultipartForm(10 << 20)
			language := r.FormValue("language")

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"text":     "Hola, mundo!",
				"language": language,
				"duration": 2.0,
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("whisper-1")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testTranscriptionAudio,
			Filename: "test.wav",
			Language: "es",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Language != "es" {
			t.Errorf("expected language es, got %s", result.Language)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"text":     "Hello, world!",
				"language": "en",
				"duration": 2.5,
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		transcriptionModel := provider.TranscriptionModel("whisper-1")

		_, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testTranscriptionAudio,
			Filename: "test.wav",
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

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}
