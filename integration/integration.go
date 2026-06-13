// Package integration provides utilities for integration testing of AI providers.
// These tests make actual API calls and require valid API keys.
package integration

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// SkipIfNoKey skips the test if the specified environment variable is not set.
func SkipIfNoKey(t *testing.T, envVar string) {
	if os.Getenv(envVar) == "" {
		t.Skipf("%s not set", envVar)
	}
}

// TestGenerateText tests basic text generation.
func TestGenerateText(t *testing.T, m stream.Model) {
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

	if result.Usage.GrandTotal() == 0 {
		t.Log("warning: usage.grandTotal is 0")
	}

	t.Logf("Generated text: %s", result.Text)
	t.Logf("Usage: %+v", result.Usage)
}

// TestGenerateTextWithSystemPrompt tests text generation with a system prompt.
func TestGenerateTextWithSystemPrompt(t *testing.T, m stream.Model) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewSystemMessage("You are a helpful assistant that always responds in pirate speak."),
			message.NewUserMessage("Hello, how are you?"),
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

// TestStreamText tests streaming text generation.
func TestStreamText(t *testing.T, m stream.Model) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := goai.StreamText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Count from 1 to 5, slowly."),
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

	text := result.Text()
	if text == "" {
		t.Error("expected non-empty final text")
	}

	t.Logf("Received %d chunks", len(chunks))
	t.Logf("Final text: %s", text)
}

// TestToolCalls tests tool calling functionality.
func TestToolCalls(t *testing.T, m stream.Model) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	calculatorSchema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"expression": {
				"type": "string",
				"description": "The mathematical expression to evaluate"
			}
		},
		"required": ["expression"]
	}`)

	tools := tool.Set{
		"calculator": {
			Name:        "calculator",
			Description: "Evaluate a mathematical expression",
			InputSchema: calculatorSchema,
			Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
				var args struct {
					Expression string `json:"expression"`
				}
				if err := json.Unmarshal(input, &args); err != nil {
					return tool.Result{}, err
				}
				// Simple evaluation - in production use a proper evaluator
				if args.Expression == "2+2" || args.Expression == "2 + 2" {
					return tool.Result{Output: "4"}, nil
				}
				return tool.Result{Output: "unknown"}, nil
			},
		},
	}

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("What is 2+2? Use the calculator tool to compute this."),
		},
		Tools: tools,
	})

	if err != nil {
		t.Fatalf("GenerateText with tools error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	} else {
		tc := result.ToolCalls[0]
		if tc.Name != "calculator" {
			t.Errorf("expected tool name 'calculator', got %s", tc.Name)
		}
		t.Logf("Tool call: %s with input %s", tc.Name, string(tc.Input))
	}

	t.Logf("Final text: %s", result.Text)
}

// TestEmbed tests single embedding generation.
func TestEmbed(t *testing.T, m model.EmbeddingModel) {
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
	t.Logf("Usage tokens: %d", result.Usage.Tokens)
}

// TestEmbedMany tests multiple embedding generation.
func TestEmbedMany(t *testing.T, m model.EmbeddingModel) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Embed(ctx, model.EmbedCallOptions{
		Values: []string{
			"First test sentence.",
			"Second test sentence.",
			"Third test sentence.",
		},
	})

	if err != nil {
		t.Fatalf("EmbedMany error: %v", err)
	}

	if len(result.Embeddings) != 3 {
		t.Errorf("expected 3 embeddings, got %d", len(result.Embeddings))
	}

	for i, emb := range result.Embeddings {
		if len(emb.Values) == 0 {
			t.Errorf("embedding %d has no values", i)
		}
	}

	t.Logf("Generated %d embeddings", len(result.Embeddings))
}

// TestGenerateImage tests image generation.
func TestGenerateImage(t *testing.T, m model.ImageModel) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.ImageCallOptions{
		Prompt: "A cute cartoon cat sitting on a windowsill",
	})

	if err != nil {
		t.Fatalf("GenerateImage error: %v", err)
	}

	if len(result.Images) == 0 {
		t.Error("expected at least one image")
	}

	// Check that we got actual image data (at least 10KB)
	if len(result.Images[0].Base64) < 10*1024 {
		t.Errorf("expected base64 data to be at least 10KB, got %d chars", len(result.Images[0].Base64))
	}

	t.Logf("Generated %d images, first image base64 size: %d chars", len(result.Images), len(result.Images[0].Base64))
}

// TestGenerateSpeech tests text-to-speech generation.
func TestGenerateSpeech(t *testing.T, m model.SpeechModel) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.SpeechCallOptions{
		Text:  "Hello, world! This is a test of text to speech.",
		Voice: "alloy",
	})

	if err != nil {
		t.Fatalf("GenerateSpeech error: %v", err)
	}

	if len(result.Audio) == 0 {
		t.Error("expected non-empty audio data")
	}

	t.Logf("Generated audio size: %d bytes", len(result.Audio))
}

// TestTranscribe tests speech-to-text transcription.
func TestTranscribe(t *testing.T, m model.TranscriptionModel, audioData []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Transcribe(ctx, model.TranscribeCallOptions{
		Audio:    audioData,
		MimeType: "audio/mp3",
	})

	if err != nil {
		t.Fatalf("Transcribe error: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty transcription text")
	}

	t.Logf("Transcription: %s", result.Text)
}

// TestImageInput tests image input for vision models.
func TestImageInput(t *testing.T, m stream.Model, imageURL string) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			{
				Role: message.RoleUser,
				Content: message.Content{
					Parts: []message.Part{
						message.TextPart{Text: "Describe the image in detail."},
						message.FilePart{Data: message.FileDataURL{URL: imageURL}, MimeType: "image/jpeg"},
					},
				},
			},
		},
	})

	if err != nil {
		t.Fatalf("GenerateText with image error: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty text")
	}

	// Check that the response mentions something about the image
	textLower := strings.ToLower(result.Text)
	if !strings.Contains(textLower, "cat") &&
		!strings.Contains(textLower, "animal") &&
		!strings.Contains(textLower, "image") {
		t.Logf("warning: response may not describe the image: %s", result.Text[:min(200, len(result.Text))])
	}

	t.Logf("Image description: %s", result.Text[:min(500, len(result.Text))])
}
