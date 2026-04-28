package anthropic

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
}

func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("claude-3-5-sonnet", func(t *testing.T) {
		m := p.LanguageModel("claude-3-5-sonnet-20241022")
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

	t.Run("claude-3-haiku", func(t *testing.T) {
		m := p.LanguageModel("claude-3-haiku-20240307")
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
	m := p.LanguageModel("claude-3-5-sonnet-20241022")

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

func TestIntegration_ToolCalls(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("claude-3-5-sonnet-20241022")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	tools := tool.Set{
		"calculator": {
			Name:        "calculator",
			Description: "Evaluate a mathematical expression",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
			Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
				return tool.Result{Output: "4"}, nil
			},
		},
	}

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("What is 2+2? Use the calculator tool."),
		},
		Tools: tools,
	})

	if err != nil {
		t.Fatalf("GenerateText with tools error: %v", err)
	}

	if len(result.ToolCalls) == 0 {
		t.Error("expected at least one tool call")
	} else {
		t.Logf("Tool call: %s with input %s", result.ToolCalls[0].Name, string(result.ToolCalls[0].Input))
	}
}

// Regression: sol passes ToolChoice as a bare string ("auto"/"required"). The
// Anthropic API rejects bare strings on tool_choice ("Input should be a valid
// dictionary or object to extract fields from"); the provider must translate
// each form into Anthropic's object shape. Hits each ai-sdk-documented form
// against the live API.
func TestIntegration_ToolChoiceForms(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("claude-haiku-4-5-20251001")

	tools := tool.Set{
		"calculator": {
			Name:        "calculator",
			Description: "Evaluate a mathematical expression",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
			Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
				return tool.Result{Output: "4"}, nil
			},
		},
	}

	cases := []struct {
		name       string
		toolChoice any
	}{
		{name: "string auto", toolChoice: "auto"},
		{name: "string required", toolChoice: "required"},
		{name: "object auto", toolChoice: map[string]any{"type": "auto"}},
		{name: "object required", toolChoice: map[string]any{"type": "required"}},
		{name: "object tool by toolName", toolChoice: map[string]any{"type": "tool", "toolName": "calculator"}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			_, err := goai.GenerateText(ctx, stream.Input{
				Model: m,
				Messages: []message.Message{
					message.NewUserMessage("What is 2+2? Use the calculator tool."),
				},
				Tools:      tools,
				ToolChoice: tc.toolChoice,
			})
			if err != nil {
				t.Fatalf("GenerateText with toolChoice=%v: %v", tc.toolChoice, err)
			}
		})
	}
}

// "none" is special — Anthropic has no native "none" tool_choice. ai-sdk
// models it by sending no tools at all, so the model must return text. Verify
// the live API accepts this against goai's translation.
func TestIntegration_ToolChoiceNone(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("claude-haiku-4-5-20251001")

	tools := tool.Set{
		"calculator": {
			Name:        "calculator",
			Description: "Evaluate a mathematical expression",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
			Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
				return tool.Result{Output: "4"}, nil
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Say hello in one word."),
		},
		Tools:      tools,
		ToolChoice: "none",
	})
	if err != nil {
		t.Fatalf("GenerateText with toolChoice=none: %v", err)
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected no tool calls with toolChoice=none, got %d", len(result.ToolCalls))
	}
	if result.Text == "" {
		t.Error("expected non-empty text response")
	}
}

func TestIntegration_ImageInput(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.LanguageModel("claude-3-5-sonnet-20241022")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	imageURL := "https://github.com/vercel/ai/blob/main/examples/ai-functions/data/comic-cat.png?raw=true"

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			{
				Role: message.RoleUser,
				Content: message.Content{
					Parts: []message.Part{
						message.TextPart{Text: "What animal is in this image? Answer in one word."},
						message.ImagePart{Image: imageURL},
					},
				},
			},
		},
	})

	if err != nil {
		t.Fatalf("GenerateText with image error: %v", err)
	}

	textLower := strings.ToLower(result.Text)
	if !strings.Contains(textLower, "cat") {
		t.Logf("warning: expected 'cat' in response, got: %s", result.Text)
	}

	t.Logf("Image description: %s", result.Text)
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
