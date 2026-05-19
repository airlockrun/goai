package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// skipIfNoKey skips the test if OPENAI_API_KEY is not set.
func skipIfNoKey(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(provider.Options{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})
}

// TestIntegration_GenerateText tests basic text generation with real API.
func TestIntegration_GenerateText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("gpt-4o-mini", func(t *testing.T) {
		testGenerateText(t, p.Chat("gpt-4o-mini"))
	})

	t.Run("gpt-3.5-turbo", func(t *testing.T) {
		testGenerateText(t, p.Chat("gpt-3.5-turbo"))
	})
}

func testGenerateText(t *testing.T, m stream.Model) {
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
		t.Log("warning: usage.totalTokens is 0")
	}

	t.Logf("Generated text: %s", result.Text)
	t.Logf("Usage: prompt=%d, completion=%d, total=%d",
		result.Usage.InputTotal(), result.Usage.OutputTotal(), result.Usage.GrandTotal())
}

// TestIntegration_GenerateTextWithSystemPrompt tests text generation with system prompt.
func TestIntegration_GenerateTextWithSystemPrompt(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Chat("gpt-4o-mini")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewSystemMessage("You are a helpful assistant that always responds in exactly 5 words."),
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
	t.Logf("Usage: %+v", result.Usage)
}

// TestIntegration_StreamText tests streaming text generation.
func TestIntegration_StreamText(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Chat("gpt-4o-mini")

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

	text := result.Text()
	if text == "" {
		t.Error("expected non-empty final text")
	}

	t.Logf("Received %d chunks", len(chunks))
	t.Logf("Final text: %s", text)
}

// TestIntegration_ToolCalls tests tool calling functionality.
func TestIntegration_ToolCalls(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Chat("gpt-4o-mini")

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
				// Simple evaluation - hardcoded for test
				if strings.Contains(args.Expression, "2") {
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

// TestIntegration_Embeddings tests embedding generation.
func TestIntegration_Embeddings(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("single embedding", func(t *testing.T) {
		m := p.EmbeddingModel("text-embedding-3-small")

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
	})

	t.Run("multiple embeddings", func(t *testing.T) {
		m := p.EmbeddingModel("text-embedding-3-small")

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
	})
}

// TestIntegration_ImageGeneration tests image generation.
func TestIntegration_ImageGeneration(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.ImageModel("dall-e-2") // dall-e-2 is cheaper for testing

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.ImageCallOptions{
		Prompt: "A simple red circle on a white background",
		Size:   "256x256",
	})

	if err != nil {
		t.Fatalf("GenerateImage error: %v", err)
	}

	if len(result.Images) == 0 {
		t.Error("expected at least one image")
	}

	// Check that we got actual image data (at least 1KB for small image)
	if len(result.Images[0].Base64) < 1024 {
		t.Errorf("expected base64 data to be at least 1KB, got %d chars", len(result.Images[0].Base64))
	}

	t.Logf("Generated %d images, first image base64 size: %d chars", len(result.Images), len(result.Images[0].Base64))
}

// TestIntegration_Speech tests text-to-speech generation.
func TestIntegration_Speech(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.SpeechModel("tts-1")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := m.Generate(ctx, model.SpeechCallOptions{
		Text:  "Hello world.",
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

// TestIntegration_ImageInput tests image input for vision models.
func TestIntegration_ImageInput(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Chat("gpt-4o-mini")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Use a publicly accessible test image
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

	if result.Text == "" {
		t.Error("expected non-empty text")
	}

	textLower := strings.ToLower(result.Text)
	if !strings.Contains(textLower, "cat") {
		t.Logf("warning: expected 'cat' in response, got: %s", result.Text)
	}

	t.Logf("Image description: %s", result.Text)
}

// TestIntegration_ErrorHandling tests error handling with invalid model.
func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Chat("no-such-model")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})

	if err == nil {
		t.Error("expected error with invalid model, got nil")
	}

	if !strings.Contains(err.Error(), "model") {
		t.Logf("Error message: %v", err)
	}
}

// TestIntegration_ResponsesAPI tests the Responses API endpoint.
func TestIntegration_ResponsesAPI(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.Responses("gpt-4o-mini")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := goai.GenerateText(ctx, stream.Input{
		Model: m,
		Messages: []message.Message{
			message.NewUserMessage("Say 'hello' and nothing else."),
		},
	})

	if err != nil {
		t.Fatalf("GenerateText via Responses API error: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty text")
	}

	t.Logf("Responses API text: %s", result.Text)
}

// generateTestPNG creates a 100x100 PNG with colored quadrants (red, green, blue, yellow).
func generateTestPNG() string {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	// Red top-left, green top-right, blue bottom-left, yellow bottom-right
	draw.Draw(img, image.Rect(0, 0, 50, 50), &image.Uniform{color.RGBA{255, 0, 0, 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(50, 0, 100, 50), &image.Uniform{color.RGBA{0, 255, 0, 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(0, 50, 50, 100), &image.Uniform{color.RGBA{0, 0, 255, 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(50, 50, 100, 100), &image.Uniform{color.RGBA{255, 255, 0, 255}}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// TestIntegration_Base64ImageResponses tests sending a base64-encoded image
// via the Responses API, simulating the attachToContext pipeline.
func TestIntegration_Base64ImageResponses(t *testing.T) {
	skipIfNoKey(t)

	b64 := generateTestPNG()

	p := getProvider()
	m := p.Responses("gpt-5.3-codex")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Step 1: Send image as a user message (direct).
	t.Run("user_message_image", func(t *testing.T) {
		result, err := goai.GenerateText(ctx, stream.Input{
			Model: m,
			Messages: []message.Message{
				{
					Role: message.RoleUser,
					Content: message.Content{
						Parts: []message.Part{
							message.TextPart{Text: "This image has 4 colored quadrants. Name the colors clockwise from top-left. Just the color names, comma-separated."},
							message.ImagePart{Image: b64, MimeType: "image/png"},
						},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		t.Logf("Response: %s", result.Text)
		lower := strings.ToLower(result.Text)
		for _, c := range []string{"red", "green", "yellow", "blue"} {
			if !strings.Contains(lower, c) {
				t.Errorf("expected %q in response", c)
			}
		}
	})

	// Step 2: Send image as a tool result attachment (simulating attachToContext).
	t.Run("tool_result_image", func(t *testing.T) {
		result, err := goai.GenerateText(ctx, stream.Input{
			Model: m,
			Messages: []message.Message{
				message.NewUserMessage("Use the run_js tool to load an image, then name the colors of the 4 quadrants clockwise from top-left. Just the color names, comma-separated."),
				{
					Role: message.RoleAssistant,
					Content: message.Content{
						Parts: []message.Part{
							message.ToolCallPart{
								ID:    "call_123",
								Name:  "run_js",
								Input: json.RawMessage(`{"code":"attachToContext(\"test.png\")"}`),
							},
						},
					},
				},
				{
					Role: message.RoleTool,
					Content: message.Content{
						Parts: []message.Part{
							message.ToolResultPart{ToolCallID: "call_123", ToolName: "run_js", Output: message.TextOutput{Value: ""}},
							message.ImagePart{Image: b64, MimeType: "image/png"},
						},
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		t.Logf("Response: %s", result.Text)
		lower := strings.ToLower(result.Text)
		for _, c := range []string{"red", "green", "yellow", "blue"} {
			if !strings.Contains(lower, c) {
				t.Errorf("expected %q in response", c)
			}
		}
	})
}
