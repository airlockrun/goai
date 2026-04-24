package hume

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/hume/src/hume-speech-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

func TestHumeProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "hume" {
		t.Errorf("expected provider ID hume, got %s", provider.ID())
	}
}

func TestHumeProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasOctave := false
	for _, m := range models {
		if strings.Contains(m, "octave") {
			hasOctave = true
		}
	}
	if !hasOctave {
		t.Error("expected 'octave' model in models list")
	}
}

func TestHumeSpeech_Generate(t *testing.T) {
	t.Run("should generate speech with required parameters", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "audio/wav")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}

		if receivedBody["text"] != "Hello from the AI SDK!" {
			t.Errorf("expected text 'Hello from the AI SDK!', got %v", receivedBody["text"])
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "audio/wav")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("X-Hume-Api-Key") != "test-api-key" {
			t.Errorf("expected X-Hume-Api-Key header, got %s", receivedHeaders.Get("X-Hume-Api-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass voice option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "audio/wav")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello from the AI SDK!",
			Voice: "test-voice",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		voice, ok := receivedBody["voice"].(map[string]any)
		if !ok {
			t.Error("expected voice object in request body")
			return
		}

		if voice["name"] != "test-voice" {
			t.Errorf("expected voice name test-voice, got %v", voice["name"])
		}
	})

	t.Run("should pass provider options", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "audio/wav")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
			ProviderOptions: map[string]any{
				"emotions": map[string]float64{
					"happiness": 0.8,
					"sadness":   0.1,
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["emotions"] == nil {
			t.Error("expected emotions in request body")
		}
	})
}

func TestHumeSpeech_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("octave")

	if speechModel.ID() != "octave" {
		t.Errorf("expected model ID octave, got %s", speechModel.ID())
	}
}

func TestHumeSpeech_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("octave")

	if speechModel.Provider() != "hume" {
		t.Errorf("expected provider hume, got %s", speechModel.Provider())
	}
}

func TestHumeSpeech_ErrorHandling(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":"Invalid API key"}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}

		if !strings.Contains(err.Error(), "Hume API error") {
			t.Errorf("expected Hume API error, got %v", err)
		}
	})
}

func TestHumeSpeech_ContentType(t *testing.T) {
	t.Run("should return content type from response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.MimeType != "audio/mpeg" {
			t.Errorf("expected mime type audio/mpeg, got %s", result.MimeType)
		}
	})

	t.Run("should default to audio/wav when no content type", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Don't set Content-Type header - Go auto-sets "application/octet-stream"
			// for binary data, which should be treated as "unknown" and default to audio/wav
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("octave")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.MimeType != "audio/wav" {
			t.Errorf("expected mime type audio/wav, got %s", result.MimeType)
		}
	})
}
