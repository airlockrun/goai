package openaicompat

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func skipIfNoConfig(t *testing.T) {
	if os.Getenv("OPENAI_COMPATIBLE_API_KEY") == "" || os.Getenv("OPENAI_COMPATIBLE_BASE_URL") == "" {
		t.Skip("OPENAI_COMPATIBLE_API_KEY or OPENAI_COMPATIBLE_BASE_URL not set")
	}
}

func getProvider() *Provider {
	return New(Options{
		APIKey:  os.Getenv("OPENAI_COMPATIBLE_API_KEY"),
		BaseURL: os.Getenv("OPENAI_COMPATIBLE_BASE_URL"),
	})
}

func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoConfig(t)

	modelID := os.Getenv("OPENAI_COMPATIBLE_MODEL")
	if modelID == "" {
		modelID = "gpt-3.5-turbo" // default model
	}

	p := getProvider()
	m := p.Model(modelID)

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
	t.Logf("Usage: %+v", result.Usage)
}

func TestIntegration_StreamText(t *testing.T) {
	skipIfNoConfig(t)

	modelID := os.Getenv("OPENAI_COMPATIBLE_MODEL")
	if modelID == "" {
		modelID = "gpt-3.5-turbo"
	}

	p := getProvider()
	m := p.Model(modelID)

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

func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoConfig(t)
	p := getProvider()
	m := p.Model("no-such-model")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})

	if err == nil {
		t.Error("expected error with invalid model")
	}
}
