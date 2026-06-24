//go:build integration

// Integration tests for the OpenRouter modality provider. These hit the live
// API and cost a small amount, so they are gated behind the `integration` build
// tag and never run in the normal suite:
//
//	go test -tags=integration ./provider/openrouter/
//
// They need OPENROUTER_API_KEY, taken from the environment or a .env file in the
// package or repo root. Models are cheap; adjust the consts if one is retired.
// A 404 "No endpoints available matching your guardrail restrictions and data
// policy" is an account setting, not a test failure — enable the needed data
// policy at https://openrouter.ai/settings/privacy (or pick a model your policy
// already permits).
package openrouter_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openrouter"
)

const (
	ttsModel = "hexgrad/kokoro-82m"
	sttModel = "openai/whisper-1"
)

func apiKey(t *testing.T) string {
	t.Helper()
	if k := os.Getenv("OPENROUTER_API_KEY"); k != "" {
		return k
	}
	for _, p := range []string{".env", "../../.env"} {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			if v, ok := strings.CutPrefix(strings.TrimSpace(line), "OPENROUTER_API_KEY="); ok {
				return strings.Trim(strings.TrimSpace(v), `"'`)
			}
		}
	}
	t.Skip("OPENROUTER_API_KEY not set (env or .env)")
	return ""
}

func TestSpeechGenerate(t *testing.T) {
	p := openrouter.New(provider.Options{APIKey: apiKey(t)})
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	res, err := p.SpeechModel(ttsModel).Generate(ctx, model.SpeechCallOptions{
		Text:  "Hello from the OpenRouter integration test.",
		Voice: "alloy",
	})
	if err != nil {
		t.Fatalf("speech generate: %v", err)
	}
	if len(res.Audio) == 0 {
		t.Fatal("expected audio bytes, got none")
	}
	t.Logf("TTS: %d bytes, %s", len(res.Audio), res.MimeType)
}

// TestTranscribeRoundTrip synthesizes a phrase then transcribes it back,
// exercising the JSON+base64 STT path (the one that differs from OpenAI).
func TestTranscribeRoundTrip(t *testing.T) {
	p := openrouter.New(provider.Options{APIKey: apiKey(t)})
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	speech, err := p.SpeechModel(ttsModel).Generate(ctx, model.SpeechCallOptions{
		Text:         "The quick brown fox jumps over the lazy dog.",
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("tts: %v", err)
	}

	tr, err := p.TranscriptionModel(sttModel).Transcribe(ctx, model.TranscribeCallOptions{
		Audio:    speech.Audio,
		MimeType: "audio/mpeg",
	})
	if err != nil {
		t.Fatalf("stt: %v", err)
	}
	if strings.TrimSpace(tr.Text) == "" {
		t.Fatal("expected transcription text, got empty")
	}
	t.Logf("STT: %q", tr.Text)
}
