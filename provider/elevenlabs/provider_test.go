package elevenlabs

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

// Translated from ai-sdk/packages/elevenlabs/src/elevenlabs-speech-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func TestElevenLabsProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "elevenlabs" {
		t.Errorf("expected provider ID elevenlabs, got %s", provider.ID())
	}
}

func TestElevenLabsProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasMultilingual := false
	for _, m := range models {
		if strings.Contains(m, "multilingual") {
			hasMultilingual = true
		}
	}
	if !hasMultilingual {
		t.Error("expected multilingual model in models list")
	}
}

func TestElevenLabsSpeech_Generate(t *testing.T) {
	t.Run("should generate speech with required parameters", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "test-voice-id",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}

		if receivedBody["text"] != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %v", receivedBody["text"])
		}

		if receivedBody["model_id"] != "eleven_multilingual_v2" {
			t.Errorf("expected model_id eleven_multilingual_v2, got %v", receivedBody["model_id"])
		}
	})

	t.Run("should pass voice settings from provider options", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "test-voice-id",
			ProviderOptions: map[string]any{
				"voice_settings": map[string]any{
					"stability":        0.5,
					"similarity_boost": 0.75,
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		voiceSettings, ok := receivedBody["voice_settings"].(map[string]any)
		if !ok {
			t.Error("expected voice_settings in request body")
			return
		}

		if voiceSettings["stability"] != 0.5 {
			t.Errorf("expected stability 0.5, got %v", voiceSettings["stability"])
		}

		if voiceSettings["similarity_boost"] != 0.75 {
			t.Errorf("expected similarity_boost 0.75, got %v", voiceSettings["similarity_boost"])
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
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "test-voice-id",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Xi-Api-Key") != "test-api-key" {
			t.Errorf("expected xi-api-key header, got %s", receivedHeaders.Get("Xi-Api-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})

	t.Run("should use default voice if not specified", func(t *testing.T) {
		var requestURL string

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestURL = r.URL.Path
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello, world!",
			// No Voice specified - should use default
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should contain a voice ID in the URL path
		if !strings.Contains(requestURL, "/text-to-speech/") {
			t.Errorf("expected /text-to-speech/ in URL path, got %s", requestURL)
		}
	})
}

func TestElevenLabsSpeech_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("eleven_multilingual_v2")

	if speechModel.ID() != "eleven_multilingual_v2" {
		t.Errorf("expected model ID eleven_multilingual_v2, got %s", speechModel.ID())
	}
}

func TestElevenLabsSpeech_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	speechModel := provider.SpeechModel("eleven_multilingual_v2")

	if speechModel.Provider() != "elevenlabs" {
		t.Errorf("expected provider elevenlabs, got %s", speechModel.Provider())
	}
}

func TestElevenLabsSpeech_ErrorHandling(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}`))
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "test-voice-id",
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestElevenLabsSpeech_MimeType(t *testing.T) {
	t.Run("should return correct mime type", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("eleven_multilingual_v2")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "test-voice-id",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.MimeType != "audio/mpeg" {
			t.Errorf("expected mime type audio/mpeg, got %s", result.MimeType)
		}
	})
}
