package bedrock

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
	if os.Getenv("AWS_ACCESS_KEY_ID") == "" || os.Getenv("AWS_SECRET_ACCESS_KEY") == "" {
		t.Skip("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{
		AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		Region:          getEnvOrDefault("AWS_REGION", "us-east-1"),
	})
}

func getEnvOrDefault(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("anthropic.claude-3-haiku-20240307-v1:0", func(t *testing.T) {
		m := p.LanguageModel("anthropic.claude-3-haiku-20240307-v1:0")
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
	})

	t.Run("amazon.titan-text-express-v1", func(t *testing.T) {
		m := p.LanguageModel("amazon.titan-text-express-v1")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		result, err := goai.GenerateText(ctx, stream.Input{
			Model: m,
			Messages: []message.Message{
				message.NewUserMessage("Say hello in exactly 3 words."),
			},
		})

		if err != nil {
			t.Fatalf("GenerateText error: %v", err)
		}

		if result.Text == "" {
			t.Error("expected non-empty text")
		}

		t.Logf("Generated text: %s", result.Text)
	})
}

func TestIntegration_StreamText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("anthropic.claude-3-haiku-20240307-v1:0")

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
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("no-such-model")

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
