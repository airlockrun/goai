package togetherai

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("TOGETHER_API_KEY") == "" {
		t.Skip("TOGETHER_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("TOGETHER_API_KEY")})
}

func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("meta-llama/Llama-3-8b-chat-hf", func(t *testing.T) {
		m := p.LanguageModel("meta-llama/Llama-3-8b-chat-hf")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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

	t.Run("mistralai/Mixtral-8x7B-Instruct-v0.1", func(t *testing.T) {
		m := p.LanguageModel("mistralai/Mixtral-8x7B-Instruct-v0.1")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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
	m := p.LanguageModel("meta-llama/Llama-3-8b-chat-hf")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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

func TestIntegration_Embeddings(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.EmbeddingModel("togethercomputer/m2-bert-80M-8k-retrieval")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Embed(ctx, model.EmbedCallOptions{
		Values: []string{"This is a test sentence for embedding."},
	})

	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}

	if len(result.Embeddings) != 1 {
		t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
	}

	if len(result.Embeddings[0].Values) == 0 {
		t.Error("expected non-empty embedding values")
	}

	t.Logf("Embedding dimension: %d", len(result.Embeddings[0].Values))
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
