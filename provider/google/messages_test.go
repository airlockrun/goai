package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Tests for message conversion - translated from ai-sdk
// Source: ai-sdk/packages/google/src/convert-to-google-generative-ai-messages.test.ts
//
// Note: goai handles message conversion inline in chat.go via
// convertToGeminiParts, convertAssistantParts, etc.

func TestConvertToGeminiParts_UserMessages(t *testing.T) {
	t.Run("should convert simple text message", func(t *testing.T) {
		content := message.Content{Text: "Hello, world!"}

		result := convertToGeminiParts(content)

		if len(result) != 1 {
			t.Fatalf("expected 1 part, got %d", len(result))
		}
		if result[0].Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got '%s'", result[0].Text)
		}
	})

	t.Run("should convert text parts", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "Part 1"},
				message.TextPart{Text: "Part 2"},
			},
		}

		result := convertToGeminiParts(content)

		if len(result) != 2 {
			t.Fatalf("expected 2 parts, got %d", len(result))
		}
		if result[0].Text != "Part 1" {
			t.Errorf("expected text 'Part 1', got '%s'", result[0].Text)
		}
		if result[1].Text != "Part 2" {
			t.Errorf("expected text 'Part 2', got '%s'", result[1].Text)
		}
	})

	t.Run("should convert image parts with inline data", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "Look at this:"},
				message.ImagePart{
					Image:    "base64encodeddata",
					MimeType: "image/png",
				},
			},
		}

		result := convertToGeminiParts(content)

		if len(result) != 2 {
			t.Fatalf("expected 2 parts, got %d", len(result))
		}

		// Check text part
		if result[0].Text != "Look at this:" {
			t.Errorf("expected text 'Look at this:', got '%s'", result[0].Text)
		}

		// Check image part
		if result[1].InlineData == nil {
			t.Fatal("expected InlineData to be set")
		}
		if result[1].InlineData.MimeType != "image/png" {
			t.Errorf("expected mimeType 'image/png', got '%s'", result[1].InlineData.MimeType)
		}
		if result[1].InlineData.Data != "base64encodeddata" {
			t.Errorf("expected data 'base64encodeddata', got '%s'", result[1].InlineData.Data)
		}
	})
}

func TestConvertAssistantParts(t *testing.T) {
	t.Run("should convert text content", func(t *testing.T) {
		content := message.Content{Text: "Hello from assistant"}

		result := convertAssistantParts(content)

		if len(result) != 1 {
			t.Fatalf("expected 1 part, got %d", len(result))
		}
		if result[0].Text != "Hello from assistant" {
			t.Errorf("expected text 'Hello from assistant', got '%s'", result[0].Text)
		}
	})

	t.Run("should convert function call parts", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.ToolCallPart{
					ID:    "call_123",
					Name:  "get_weather",
					Input: json.RawMessage(`{"location": "NYC"}`),
				},
			},
		}

		result := convertAssistantParts(content)

		if len(result) != 1 {
			t.Fatalf("expected 1 part, got %d", len(result))
		}
		if result[0].FunctionCall == nil {
			t.Fatal("expected FunctionCall to be set")
		}
		if result[0].FunctionCall.Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", result[0].FunctionCall.Name)
		}
		if result[0].FunctionCall.Args["location"] != "NYC" {
			t.Errorf("expected location 'NYC', got '%v'", result[0].FunctionCall.Args["location"])
		}
	})

	t.Run("should convert text with function call", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "Let me check the weather"},
				message.ToolCallPart{
					ID:    "call_456",
					Name:  "get_weather",
					Input: json.RawMessage(`{"location": "SF"}`),
				},
			},
		}

		result := convertAssistantParts(content)

		if len(result) != 2 {
			t.Fatalf("expected 2 parts, got %d", len(result))
		}

		// Check text part
		if result[0].Text != "Let me check the weather" {
			t.Errorf("expected text 'Let me check the weather', got '%s'", result[0].Text)
		}

		// Check function call part
		if result[1].FunctionCall == nil {
			t.Fatal("expected FunctionCall to be set")
		}
		if result[1].FunctionCall.Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", result[1].FunctionCall.Name)
		}
	})
}

// Tool result conversion tests - translated from ai-sdk
// Source: ai-sdk/packages/google/src/convert-to-google-generative-ai-messages.test.ts

func TestConvertToolMessages_Google(t *testing.T) {
	t.Run("should convert simple tool result", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Result:     "72°F",
					},
				},
			},
		}

		// Simulate what buildRequest does for RoleTool
		var parts []geminiPart
		for _, part := range msg.Content.Parts {
			switch p := part.(type) {
			case message.ToolResultPart:
				parts = append(parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name:     p.ToolName,
						Response: map[string]any{"result": p.Result},
					},
				})
			}
		}

		if len(parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(parts))
		}
		if parts[0].FunctionResponse == nil {
			t.Fatal("expected FunctionResponse to be set")
		}
		if parts[0].FunctionResponse.Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", parts[0].FunctionResponse.Name)
		}
	})

	t.Run("should convert tool result with image as inlineData", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "call_123",
						ToolName:   "screenshot",
						Result:     "screenshot taken",
					},
					message.ImagePart{
						Image:    "iVBORw0KGgo=",
						MimeType: "image/png",
					},
				},
			},
		}

		// Simulate what buildRequest does for RoleTool
		var parts []geminiPart
		for _, part := range msg.Content.Parts {
			switch p := part.(type) {
			case message.ToolResultPart:
				parts = append(parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name:     p.ToolName,
						Response: map[string]any{"result": p.Result},
					},
				})
			case message.ImagePart:
				parts = append(parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: p.MimeType,
						Data:     p.Image,
					},
				})
				parts = append(parts, geminiPart{
					Text: "Tool executed successfully and returned this image as a response",
				})
			}
		}

		if len(parts) != 3 {
			t.Fatalf("expected 3 parts, got %d", len(parts))
		}

		// Part 1: function response
		if parts[0].FunctionResponse == nil {
			t.Fatal("expected FunctionResponse to be set")
		}

		// Part 2: inline data
		if parts[1].InlineData == nil {
			t.Fatal("expected InlineData to be set")
		}
		if parts[1].InlineData.MimeType != "image/png" {
			t.Errorf("expected mimeType 'image/png', got '%s'", parts[1].InlineData.MimeType)
		}
		if parts[1].InlineData.Data != "iVBORw0KGgo=" {
			t.Errorf("expected data 'iVBORw0KGgo=', got '%s'", parts[1].InlineData.Data)
		}

		// Part 3: synthetic text
		if parts[2].Text != "Tool executed successfully and returned this image as a response" {
			t.Errorf("expected synthetic text, got '%s'", parts[2].Text)
		}
	})
}

// Tool conversion tests are in tools_test.go

func TestGetTextFromContent_Google(t *testing.T) {
	t.Run("should get text from Text field", func(t *testing.T) {
		content := message.Content{Text: "Hello"}

		result := getTextFromContent(content)

		if result != "Hello" {
			t.Errorf("expected 'Hello', got '%s'", result)
		}
	})

	t.Run("should get text from TextPart", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "From part"},
			},
		}

		result := getTextFromContent(content)

		if result != "From part" {
			t.Errorf("expected 'From part', got '%s'", result)
		}
	})

	t.Run("should return empty string for no text", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.ImagePart{Image: "data"},
			},
		}

		result := getTextFromContent(content)

		if result != "" {
			t.Errorf("expected empty string, got '%s'", result)
		}
	})
}

// Verifies convertToGeminiParts emits fileData (not inlineData) when a
// FilePart carries a URL or an ImagePart's Image is http/https — matches
// ai-sdk's convert-to-google-generative-ai-messages.ts handling of
// URL-backed file parts.
func TestConvertToGeminiParts_URLBacked(t *testing.T) {
	t.Run("ImagePart with https URL becomes fileData", func(t *testing.T) {
		parts := convertToGeminiParts(message.Content{Parts: []message.Part{
			message.ImagePart{Image: "https://example.com/cat.png", MimeType: "image/png"},
		}})
		if len(parts) != 1 || parts[0].FileData == nil {
			t.Fatalf("expected single fileData part, got %+v", parts)
		}
		if parts[0].FileData.FileURI != "https://example.com/cat.png" {
			t.Errorf("FileURI = %q", parts[0].FileData.FileURI)
		}
		if parts[0].InlineData != nil {
			t.Error("InlineData should be nil when URL was provided")
		}
	})

	t.Run("ImagePart with base64 stays as inlineData", func(t *testing.T) {
		parts := convertToGeminiParts(message.Content{Parts: []message.Part{
			message.ImagePart{Image: "base64data", MimeType: "image/png"},
		}})
		if parts[0].InlineData == nil || parts[0].FileData != nil {
			t.Errorf("expected inlineData only, got %+v", parts[0])
		}
	})

	t.Run("FilePart.URL becomes fileData", func(t *testing.T) {
		parts := convertToGeminiParts(message.Content{Parts: []message.Part{
			message.FilePart{URL: "gs://bucket/doc.pdf", MimeType: "application/pdf"},
		}})
		if parts[0].FileData == nil || parts[0].FileData.FileURI != "gs://bucket/doc.pdf" {
			t.Errorf("FileData = %+v", parts[0].FileData)
		}
	})

	t.Run("FilePart.Data becomes inlineData", func(t *testing.T) {
		parts := convertToGeminiParts(message.Content{Parts: []message.Part{
			message.FilePart{Data: "b64", MimeType: "application/pdf"},
		}})
		if parts[0].InlineData == nil || parts[0].InlineData.Data != "b64" {
			t.Errorf("InlineData = %+v", parts[0].InlineData)
		}
	})
}

// Gemini tool messages with mixed parts should emit functionResponse
// plus sibling fileData/inlineData parts in one user turn — matches the
// multimodal functionResponse behavior (ai-sdk #47114a3).
func TestGeminiRequest_ToolResultMultipart(t *testing.T) {
	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	p := New(Options{APIKey: "test", BaseURL: server.URL})
	m := p.Model("gemini-2.5-pro")

	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("look at this"),
			message.NewAssistantMessageWithParts(
				message.ToolCallPart{ID: "call-1", Name: "fetch_report", Input: json.RawMessage(`{}`)},
			),
			{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "call-1", ToolName: "fetch_report", Result: "the-summary"},
					message.FilePart{URL: "gs://bucket/report.pdf", MimeType: "application/pdf"},
				}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}

	if len(capturedBody) == 0 {
		t.Fatal("no request body captured")
	}

	var parsed struct {
		Contents []struct {
			Role  string       `json:"role,omitempty"`
			Parts []geminiPart `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(capturedBody, &parsed); err != nil {
		t.Fatalf("parse request body: %v", err)
	}

	// Last content block is the tool-result turn with functionResponse +
	// fileData parts side by side.
	last := parsed.Contents[len(parsed.Contents)-1]
	if last.Role != "user" {
		t.Errorf("tool turn role = %q, want user", last.Role)
	}
	if len(last.Parts) != 2 {
		t.Fatalf("expected 2 parts (functionResponse + fileData), got %d: %+v", len(last.Parts), last.Parts)
	}
	if last.Parts[0].FunctionResponse == nil {
		t.Errorf("parts[0] should be functionResponse, got %+v", last.Parts[0])
	}
	if last.Parts[1].FileData == nil || last.Parts[1].FileData.FileURI != "gs://bucket/report.pdf" {
		t.Errorf("parts[1] should be fileData with gs:// URI, got %+v", last.Parts[1])
	}
}
