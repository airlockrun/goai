package lmnt

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/lmnt/src/lmnt-speech-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

func TestLMNTProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "lmnt" {
		t.Errorf("expected provider ID lmnt, got %s", provider.ID())
	}
}

func TestLMNTProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasLily := false
	for _, m := range models {
		if m == "lily" {
			hasLily = true
		}
	}
	if !hasLily {
		t.Error("expected 'lily' model in models list")
	}
}

func TestLMNTSpeech_Generate(t *testing.T) {
	t.Run("should generate speech with required parameters", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("lily")

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

		if receivedBody["voice"] != "lily" {
			t.Errorf("expected voice lily, got %v", receivedBody["voice"])
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
		speechModel := provider.SpeechModel("lily")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("X-Api-Key") != "test-api-key" {
			t.Errorf("expected X-Api-Key header, got %s", receivedHeaders.Get("X-Api-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should pass speed option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("lily")

		speed := 1.5
		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello from the AI SDK!",
			Speed: &speed,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["speed"] != 1.5 {
			t.Errorf("expected speed 1.5, got %v", receivedBody["speed"])
		}
	})

	t.Run("should pass output format", func(t *testing.T) {
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
		speechModel := provider.SpeechModel("lily")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:         "Hello from the AI SDK!",
			OutputFormat: "wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["format"] != "wav" {
			t.Errorf("expected format wav, got %v", receivedBody["format"])
		}

		if result.MimeType != "audio/wav" {
			t.Errorf("expected mime type audio/wav, got %s", result.MimeType)
		}
	})
}

func TestLMNTSpeech_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("lily")

	if speechModel.ID() != "lily" {
		t.Errorf("expected model ID lily, got %s", speechModel.ID())
	}
}

func TestLMNTSpeech_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("lily")

	if speechModel.Provider() != "lmnt" {
		t.Errorf("expected provider lmnt, got %s", speechModel.Provider())
	}
}

func TestLMNTSpeech_ErrorHandling(t *testing.T) {
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
		speechModel := provider.SpeechModel("lily")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello from the AI SDK!",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}

		if !strings.Contains(err.Error(), "LMNT API error") {
			t.Errorf("expected LMNT API error, got %v", err)
		}
	})
}

func TestLMNTSpeech_MimeType(t *testing.T) {
	formats := []struct {
		format   string
		mimeType string
	}{
		{"mp3", "audio/mpeg"},
		{"wav", "audio/wav"},
		{"aac", "audio/aac"},
	}

	for _, tc := range formats {
		t.Run("should return correct mime type for "+tc.format, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.mimeType)
				w.Write(testAudioData)
			}))
			defer server.Close()

			provider := New(Options{
				APIKey:  "test-api-key",
				BaseURL: server.URL,
			})
			speechModel := provider.SpeechModel("lily")

			result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
				Text:         "Hello",
				OutputFormat: tc.format,
			})

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.MimeType != tc.mimeType {
				t.Errorf("expected mime type %s, got %s", tc.mimeType, result.MimeType)
			}
		})
	}
}
