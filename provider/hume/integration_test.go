package hume

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("HUME_API_KEY") == "" {
		t.Skip("HUME_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("HUME_API_KEY")})
}

func TestIntegration_Speech(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.SpeechModel("octave")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.SpeechCallOptions{
		Text: "Hello world. This is a test of text to speech with emotional expression.",
	})

	if err != nil {
		t.Fatalf("Generate speech error: %v", err)
	}

	if len(result.Audio) == 0 {
		t.Error("expected non-empty audio data")
	}

	t.Logf("Generated audio size: %d bytes", len(result.Audio))
}

func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.SpeechModel("invalid-model")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := m.Generate(ctx, model.SpeechCallOptions{
		Text: "Hello",
	})

	if err == nil {
		t.Error("expected error with invalid model")
	}
}
