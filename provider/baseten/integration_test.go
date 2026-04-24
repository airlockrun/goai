package baseten

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("BASETEN_API_KEY") == "" {
		t.Skip("BASETEN_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("BASETEN_API_KEY")})
}

func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoKey(t)

	// Baseten requires a specific model deployment ID
	modelID := os.Getenv("BASETEN_MODEL_ID")
	if modelID == "" {
		t.Skip("BASETEN_MODEL_ID not set - Baseten requires a deployed model ID")
	}

	p := getProvider()
	m := p.LanguageModel(modelID)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Write a haiku about programming. Just output the haiku, nothing else."),
		},
	})

	if err != nil {
		t.Fatalf("GenerateText error: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty text")
	}

	t.Logf("Generated text: %s", result.Text)
}

func TestIntegration_StreamText(t *testing.T) {
	skipIfNoKey(t)

	modelID := os.Getenv("BASETEN_MODEL_ID")
	if modelID == "" {
		t.Skip("BASETEN_MODEL_ID not set")
	}

	p := getProvider()
	m := p.LanguageModel(modelID)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := goai.StreamText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Count from 1 to 5, one number per line."),
		},
	})

	if err != nil {
		t.Fatalf("StreamText error: %v", err)
	}

	var chunks []string
	for event := range result.FullStream {
		if event.Type == stream.EventTextDelta {
			if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
				chunks = append(chunks, delta.Text)
			}
		}
	}

	if len(chunks) == 0 {
		t.Error("expected at least one text chunk")
	}

	t.Logf("Received %d chunks", len(chunks))
	t.Logf("Final text: %s", result.Text())
}
