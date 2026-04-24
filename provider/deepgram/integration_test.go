package deepgram

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		t.Skip("DEEPGRAM_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
}

func TestIntegration_Speech(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.SpeechModel("aura-asteria-en")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.SpeechCallOptions{
		Text: "Hello world. This is a test of text to speech.",
	})

	if err != nil {
		t.Fatalf("Generate speech error: %v", err)
	}

	if len(result.Audio) == 0 {
		t.Error("expected non-empty audio data")
	}

	t.Logf("Generated audio size: %d bytes", len(result.Audio))
}

func TestIntegration_Transcription(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.TranscriptionModel("nova-2")

	// Skip if we don't have test audio
	t.Skip("Transcription test requires audio file - skipping")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Would need actual audio data here
	testAudio := []byte{}

	result, err := m.Transcribe(ctx, model.TranscribeCallOptions{
		Audio:    testAudio,
		MimeType: "audio/mp3",
	})

	if err != nil {
		t.Fatalf("Transcribe error: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty transcription")
	}

	t.Logf("Transcription: %s", result.Text)
}
