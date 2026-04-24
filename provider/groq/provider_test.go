package groq

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
)

// Translated from ai-sdk/packages/groq/src/groq-transcription-model.test.ts

var testTranscriptionAudio = []byte{1, 2, 3, 4, 5, 6, 7, 8}

func TestGroqProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "groq" {
		t.Errorf("expected provider ID groq, got %s", provider.ID())
	}
}

func TestGroqProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	// Check for some known models
	hasLlama := false
	for _, m := range models {
		if m == "llama-3.3-70b-versatile" {
			hasLlama = true
		}
	}
	if !hasLlama {
		t.Error("expected llama-3.3-70b-versatile in models list")
	}
}

func TestGroqTranscription_DoTranscribe(t *testing.T) {
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

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("whisper-large-v3")

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

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		transcriptionModel := provider.TranscriptionModel("whisper-large-v3")

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

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		transcriptionModel := provider.TranscriptionModel("whisper-large-v3")

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

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestGroqTranscription_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.TranscriptionModel("whisper-large-v3")

	if model.ID() != "whisper-large-v3" {
		t.Errorf("expected model ID whisper-large-v3, got %s", model.ID())
	}
}

func TestGroqTranscription_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.TranscriptionModel("whisper-large-v3")

	if model.Provider() != "groq" {
		t.Errorf("expected provider groq, got %s", model.Provider())
	}
}

// Tests for ProviderOptions - verifies groqRequestModifier wires up options correctly

func TestGroqRequestModifier_ReasoningEffort(t *testing.T) {
	providerOptions := map[string]any{
		"reasoningEffort": "high",
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["reasoning_effort"] != "high" {
		t.Errorf("expected reasoning_effort 'high', got %v", extra["reasoning_effort"])
	}
}

func TestGroqRequestModifier_ReasoningFormat(t *testing.T) {
	providerOptions := map[string]any{
		"reasoningFormat": "parsed",
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["reasoning_format"] != "parsed" {
		t.Errorf("expected reasoning_format 'parsed', got %v", extra["reasoning_format"])
	}
}

func TestGroqRequestModifier_ServiceTier(t *testing.T) {
	providerOptions := map[string]any{
		"serviceTier": "flex",
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["service_tier"] != "flex" {
		t.Errorf("expected service_tier 'flex', got %v", extra["service_tier"])
	}
}

func TestGroqRequestModifier_User(t *testing.T) {
	providerOptions := map[string]any{
		"user": "user_123",
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["user"] != "user_123" {
		t.Errorf("expected user 'user_123', got %v", extra["user"])
	}
}

func TestGroqRequestModifier_ParallelToolCalls(t *testing.T) {
	providerOptions := map[string]any{
		"parallelToolCalls": false,
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["parallel_tool_calls"] != false {
		t.Errorf("expected parallel_tool_calls false, got %v", extra["parallel_tool_calls"])
	}
}

func TestGroqRequestModifier_AllOptions(t *testing.T) {
	providerOptions := map[string]any{
		"reasoningEffort":   "medium",
		"reasoningFormat":   "raw",
		"serviceTier":       "on_demand",
		"user":              "test_user",
		"parallelToolCalls": true,
	}

	extra, _, err := groqRequestModifier(providerOptions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if extra["reasoning_effort"] != "medium" {
		t.Errorf("expected reasoning_effort 'medium', got %v", extra["reasoning_effort"])
	}
	if extra["reasoning_format"] != "raw" {
		t.Errorf("expected reasoning_format 'raw', got %v", extra["reasoning_format"])
	}
	if extra["service_tier"] != "on_demand" {
		t.Errorf("expected service_tier 'on_demand', got %v", extra["service_tier"])
	}
	if extra["user"] != "test_user" {
		t.Errorf("expected user 'test_user', got %v", extra["user"])
	}
	if extra["parallel_tool_calls"] != true {
		t.Errorf("expected parallel_tool_calls true, got %v", extra["parallel_tool_calls"])
	}
}
