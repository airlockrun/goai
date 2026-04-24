package assemblyai

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("ASSEMBLYAI_API_KEY") == "" {
		t.Skip("ASSEMBLYAI_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("ASSEMBLYAI_API_KEY")})
}

func TestIntegration_Transcription(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.TranscriptionModel("best")

	// Skip if we don't have test audio - AssemblyAI requires actual audio
	t.Skip("Transcription test requires audio file - skipping")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
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

func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoKey(t)
	// AssemblyAI doesn't have model selection in the same way
	t.Skip("AssemblyAI uses async API - skipping error test")
}
