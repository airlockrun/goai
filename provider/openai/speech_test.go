package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
)

// Translated from ai-sdk/packages/openai/src/speech/openai-speech-model.test.ts

var testAudioData = []byte{1, 2, 3, 4, 5, 6, 7, 8}

func TestOpenAISpeech_DoGenerate(t *testing.T) {
	t.Run("should generate speech", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("tts-1")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "alloy",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}

		for i, v := range result.Audio {
			if v != testAudioData[i] {
				t.Errorf("expected audio[%d] = %d, got %d", i, testAudioData[i], v)
			}
		}
	})

	t.Run("should pass voice and speed", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		speechModel := provider.SpeechModel("tts-1")

		speed := 1.5
		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "nova",
			Speed: &speed,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["voice"] != "nova" {
			t.Errorf("expected voice nova, got %v", receivedBody["voice"])
		}

		if receivedBody["speed"] != 1.5 {
			t.Errorf("expected speed 1.5, got %v", receivedBody["speed"])
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

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		speechModel := provider.SpeechModel("tts-1")

		_, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "alloy",
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

func TestOpenAISpeech_Warnings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write(testAudioData)
	}))
	defer srv.Close()

	p := New(provider.Options{APIKey: "k", BaseURL: srv.URL})
	sm := p.SpeechModel("tts-1")

	t.Run("warns on unsupported outputFormat", func(t *testing.T) {
		res, err := sm.Generate(context.Background(), model.SpeechCallOptions{
			Text:         "hi",
			OutputFormat: "ogg",
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		testutil.AssertResultWarning(t, res.Warnings, stream.Warning{
			Type:    stream.WarningUnsupported,
			Feature: "outputFormat",
		})
	})

	t.Run("no warning for supported outputFormat", func(t *testing.T) {
		res, err := sm.Generate(context.Background(), model.SpeechCallOptions{
			Text:         "hi",
			OutputFormat: "wav",
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(res.Warnings) != 0 {
			t.Errorf("expected no warnings, got %v", res.Warnings)
		}
	})
}
