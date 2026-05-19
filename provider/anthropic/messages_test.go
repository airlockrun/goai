package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/message"
)

// Tests for message conversion - translated from ai-sdk
// Source: ai-sdk/packages/anthropic/src/convert-to-anthropic-messages-prompt.test.ts
//
// Note: goai handles message conversion inline in chat.go via
// convertToAnthropicContent, convertAssistantContent, etc.

func TestConvertToAnthropicContent_UserMessages(t *testing.T) {
	t.Run("should convert simple text message", func(t *testing.T) {
		content := message.Content{Text: "Hello, world!"}

		result := convertToAnthropicContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "text" {
			t.Errorf("expected type 'text', got '%s'", result[0].Type)
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

		result := convertToAnthropicContent(content, nil)

		if len(result) != 2 {
			t.Fatalf("expected 2 blocks, got %d", len(result))
		}
		if result[0].Text != "Part 1" {
			t.Errorf("expected text 'Part 1', got '%s'", result[0].Text)
		}
		if result[1].Text != "Part 2" {
			t.Errorf("expected text 'Part 2', got '%s'", result[1].Text)
		}
	})

	t.Run("should convert image parts", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "Look at this:"},
				message.ImagePart{
					Image:    "base64encodeddata",
					MimeType: "image/png",
				},
			},
		}

		result := convertToAnthropicContent(content, nil)

		if len(result) != 2 {
			t.Fatalf("expected 2 blocks, got %d", len(result))
		}

		// Check text block
		if result[0].Type != "text" {
			t.Errorf("expected type 'text', got '%s'", result[0].Type)
		}

		// Check image block
		if result[1].Type != "image" {
			t.Errorf("expected type 'image', got '%s'", result[1].Type)
		}
		if result[1].Source == nil {
			t.Fatal("expected source to be set")
		}
		if result[1].Source.Type != "base64" {
			t.Errorf("expected source type 'base64', got '%s'", result[1].Source.Type)
		}
		if result[1].Source.MediaType != "image/png" {
			t.Errorf("expected media type 'image/png', got '%s'", result[1].Source.MediaType)
		}
		if result[1].Source.Data != "base64encodeddata" {
			t.Errorf("expected data 'base64encodeddata', got '%s'", result[1].Source.Data)
		}
	})

	t.Run("should convert PDF file part to document block", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.FilePart{
					Data:     "cGRmZGF0YQ==",
					MimeType: "application/pdf",
					Filename: "report.pdf",
				},
			},
		}

		result := convertToAnthropicContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "document" {
			t.Errorf("expected type 'document', got '%s'", result[0].Type)
		}
		if result[0].Source == nil {
			t.Fatal("expected source to be set")
		}
		if result[0].Source.Type != "base64" {
			t.Errorf("expected source type 'base64', got '%s'", result[0].Source.Type)
		}
		if result[0].Source.MediaType != "application/pdf" {
			t.Errorf("expected media type 'application/pdf', got '%s'", result[0].Source.MediaType)
		}
		if result[0].Title != "report.pdf" {
			t.Errorf("expected title 'report.pdf', got '%s'", result[0].Title)
		}

		// Verify JSON serialization
		data, _ := json.Marshal(result[0])
		var m map[string]any
		json.Unmarshal(data, &m)
		if m["type"] != "document" {
			t.Errorf("JSON type should be 'document'")
		}
	})

	t.Run("should convert text/plain file part to text document block", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.FilePart{
					Data:     "Hello, this is a text document.",
					MimeType: "text/plain",
					Filename: "notes.txt",
				},
			},
		}

		result := convertToAnthropicContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "document" {
			t.Errorf("expected type 'document', got '%s'", result[0].Type)
		}
		if result[0].Source.Type != "text" {
			t.Errorf("expected source type 'text', got '%s'", result[0].Source.Type)
		}
		if result[0].Source.MediaType != "text/plain" {
			t.Errorf("expected media type 'text/plain', got '%s'", result[0].Source.MediaType)
		}
		if result[0].Title != "notes.txt" {
			t.Errorf("expected title 'notes.txt', got '%s'", result[0].Title)
		}
	})

	t.Run("should convert image file part to image block", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.FilePart{
					Data:     "aW1hZ2VkYXRh",
					MimeType: "image/png",
					Filename: "photo.png",
				},
			},
		}

		result := convertToAnthropicContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "image" {
			t.Errorf("expected type 'image', got '%s'", result[0].Type)
		}
		if result[0].Source.MediaType != "image/png" {
			t.Errorf("expected media type 'image/png', got '%s'", result[0].Source.MediaType)
		}
	})
}

func TestConvertAssistantContent(t *testing.T) {
	t.Run("should convert text content", func(t *testing.T) {
		content := message.Content{Text: "Hello from assistant"}

		result := convertAssistantContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "text" {
			t.Errorf("expected type 'text', got '%s'", result[0].Type)
		}
		if result[0].Text != "Hello from assistant" {
			t.Errorf("expected text 'Hello from assistant', got '%s'", result[0].Text)
		}
	})

	t.Run("should convert tool use blocks", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.ToolCallPart{
					ID:    "toolu_123",
					Name:  "get_weather",
					Input: json.RawMessage(`{"location": "NYC"}`),
				},
			},
		}

		result := convertAssistantContent(content, nil)

		if len(result) != 1 {
			t.Fatalf("expected 1 block, got %d", len(result))
		}
		if result[0].Type != "tool_use" {
			t.Errorf("expected type 'tool_use', got '%s'", result[0].Type)
		}
		if result[0].ID != "toolu_123" {
			t.Errorf("expected ID 'toolu_123', got '%s'", result[0].ID)
		}
		if result[0].Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", result[0].Name)
		}
	})

	t.Run("should convert text with tool use", func(t *testing.T) {
		content := message.Content{
			Parts: []message.Part{
				message.TextPart{Text: "Let me check the weather"},
				message.ToolCallPart{
					ID:    "toolu_456",
					Name:  "get_weather",
					Input: json.RawMessage(`{"location": "SF"}`),
				},
			},
		}

		result := convertAssistantContent(content, nil)

		if len(result) != 2 {
			t.Fatalf("expected 2 blocks, got %d", len(result))
		}

		// Check text block
		if result[0].Type != "text" {
			t.Errorf("expected type 'text', got '%s'", result[0].Type)
		}
		if result[0].Text != "Let me check the weather" {
			t.Errorf("expected text 'Let me check the weather', got '%s'", result[0].Text)
		}

		// Check tool_use block
		if result[1].Type != "tool_use" {
			t.Errorf("expected type 'tool_use', got '%s'", result[1].Type)
		}
		if result[1].Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", result[1].Name)
		}
	})
}

// Tool result conversion tests - translated from ai-sdk
// Source: ai-sdk/packages/anthropic/src/convert-to-anthropic-messages-prompt.test.ts

func TestConvertToolMessages(t *testing.T) {
	t.Run("should convert simple tool result", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_123",
						ToolName:   "get_weather",
						Output:     message.TextOutput{Value: "72°F and sunny"},
					},
				},
			},
		}

		result := convertToolMessages(msg)

		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		if result[0].Role != "user" {
			t.Errorf("expected role 'user', got '%s'", result[0].Role)
		}
		if len(result[0].Content) != 1 {
			t.Fatalf("expected 1 content block, got %d", len(result[0].Content))
		}
		block := result[0].Content[0]
		if block.Type != "tool_result" {
			t.Errorf("expected type 'tool_result', got '%s'", block.Type)
		}
		if block.ToolUseID != "toolu_123" {
			t.Errorf("expected tool_use_id 'toolu_123', got '%s'", block.ToolUseID)
		}
		if block.Content != "72°F and sunny" {
			t.Errorf("expected content '72°F and sunny', got '%v'", block.Content)
		}
	})

	t.Run("should handle tool result with content parts", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_123",
						ToolName:   "screenshot",
						Output:     message.TextOutput{Value: "Here is the screenshot"},
					},
					message.ImagePart{
						Image:    "iVBORw0KGgo=",
						MimeType: "image/png",
					},
				},
			},
		}

		result := convertToolMessages(msg)

		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}
		block := result[0].Content[0]
		if block.Type != "tool_result" {
			t.Errorf("expected type 'tool_result', got '%s'", block.Type)
		}

		// Content should be an array of blocks
		contentBlocks, ok := block.Content.([]anthropicContentBlock)
		if !ok {
			t.Fatalf("expected content to be []anthropicContentBlock, got %T", block.Content)
		}
		if len(contentBlocks) != 2 {
			t.Fatalf("expected 2 content blocks, got %d", len(contentBlocks))
		}

		// First block: text result
		if contentBlocks[0].Type != "text" {
			t.Errorf("expected first block type 'text', got '%s'", contentBlocks[0].Type)
		}
		if contentBlocks[0].Text != "Here is the screenshot" {
			t.Errorf("expected text 'Here is the screenshot', got '%s'", contentBlocks[0].Text)
		}

		// Second block: image
		if contentBlocks[1].Type != "image" {
			t.Errorf("expected second block type 'image', got '%s'", contentBlocks[1].Type)
		}
		if contentBlocks[1].Source == nil {
			t.Fatal("expected source to be set")
		}
		if contentBlocks[1].Source.Type != "base64" {
			t.Errorf("expected source type 'base64', got '%s'", contentBlocks[1].Source.Type)
		}
		if contentBlocks[1].Source.MediaType != "image/png" {
			t.Errorf("expected media type 'image/png', got '%s'", contentBlocks[1].Source.MediaType)
		}
		if contentBlocks[1].Source.Data != "iVBORw0KGgo=" {
			t.Errorf("expected data 'iVBORw0KGgo=', got '%s'", contentBlocks[1].Source.Data)
		}
	})

	t.Run("should handle tool result with PDF content", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_456",
						ToolName:   "read_pdf",
						Output:     message.TextOutput{Value: "PDF contents"},
					},
					message.FilePart{
						Data:     "JVBERi0xLjQ=",
						MimeType: "application/pdf",
						Filename: "report.pdf",
					},
				},
			},
		}

		result := convertToolMessages(msg)

		block := result[0].Content[0]
		contentBlocks, ok := block.Content.([]anthropicContentBlock)
		if !ok {
			t.Fatalf("expected content to be []anthropicContentBlock, got %T", block.Content)
		}
		if len(contentBlocks) != 2 {
			t.Fatalf("expected 2 content blocks, got %d", len(contentBlocks))
		}

		// Second block: document
		if contentBlocks[1].Type != "document" {
			t.Errorf("expected type 'document', got '%s'", contentBlocks[1].Type)
		}
		if contentBlocks[1].Source.MediaType != "application/pdf" {
			t.Errorf("expected media type 'application/pdf', got '%s'", contentBlocks[1].Source.MediaType)
		}
		if contentBlocks[1].Title != "report.pdf" {
			t.Errorf("expected title 'report.pdf', got '%s'", contentBlocks[1].Title)
		}
	})

	t.Run("should handle tool result with mixed content", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_789",
						ToolName:   "analyze",
						Output:     message.TextOutput{Value: "Analysis complete"},
					},
					message.TextPart{Text: "Additional context"},
					message.ImagePart{
						Image:    "base64data",
						MimeType: "image/jpeg",
					},
				},
			},
		}

		result := convertToolMessages(msg)

		block := result[0].Content[0]
		contentBlocks, ok := block.Content.([]anthropicContentBlock)
		if !ok {
			t.Fatalf("expected content to be []anthropicContentBlock, got %T", block.Content)
		}
		// text result + TextPart + ImagePart = 3 blocks
		if len(contentBlocks) != 3 {
			t.Fatalf("expected 3 content blocks, got %d", len(contentBlocks))
		}
		if contentBlocks[0].Type != "text" {
			t.Errorf("expected first block type 'text', got '%s'", contentBlocks[0].Type)
		}
		if contentBlocks[1].Type != "text" {
			t.Errorf("expected second block type 'text', got '%s'", contentBlocks[1].Type)
		}
		if contentBlocks[2].Type != "image" {
			t.Errorf("expected third block type 'image', got '%s'", contentBlocks[2].Type)
		}
	})

	t.Run("should propagate IsError", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_err",
						ToolName:   "failing_tool",
						Output:     message.ErrorTextOutput{Value: "Something went wrong"},
					},
				},
			},
		}

		result := convertToolMessages(msg)

		block := result[0].Content[0]
		if !block.IsError {
			t.Error("expected is_error to be true")
		}
	})

	t.Run("should serialize correctly to JSON", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{
				Parts: []message.Part{
					message.ToolResultPart{
						ToolCallID: "toolu_123",
						ToolName:   "screenshot",
						Output:     message.TextOutput{Value: "Result text"},
					},
					message.ImagePart{
						Image:    "aW1hZ2U=",
						MimeType: "image/png",
					},
				},
			},
		}

		result := convertToolMessages(msg)
		data, err := json.Marshal(result[0])
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var raw map[string]any
		json.Unmarshal(data, &raw)
		content := raw["content"].([]any)
		toolResult := content[0].(map[string]any)

		// The tool_result block should have a content array
		contentArr := toolResult["content"].([]any)
		if len(contentArr) != 2 {
			t.Fatalf("expected 2 content items in JSON, got %d", len(contentArr))
		}

		textBlock := contentArr[0].(map[string]any)
		if textBlock["type"] != "text" {
			t.Errorf("expected type 'text', got '%v'", textBlock["type"])
		}
		imageBlock := contentArr[1].(map[string]any)
		if imageBlock["type"] != "image" {
			t.Errorf("expected type 'image', got '%v'", imageBlock["type"])
		}
	})
}

// TestConvertToolMessages_OutputVariantWire asserts the Anthropic wire
// mapping per ToolResultOutput variant (ADR §8): is_error is set ONLY for
// error-text/error-json; text/json/execution-denied/content never set it;
// execution-denied serializes to its reason string; and a ContentOutput
// with an image item produces an Anthropic image block.
func TestConvertToolMessages_OutputVariantWire(t *testing.T) {
	t.Run("is_error only for error variants", func(t *testing.T) {
		cases := []struct {
			name     string
			output   message.ToolResultOutput
			wantErr  bool
			wantWire string // expected scalar block.Content (when not a block array)
		}{
			{"text", message.TextOutput{Value: "hello"}, false, "hello"},
			{"json", message.JSONOutput{Value: map[string]any{"k": "v"}}, false, `{"k":"v"}`},
			{"error-text", message.ErrorTextOutput{Value: "boom"}, true, "boom"},
			{"error-json", message.ErrorJSONOutput{Value: map[string]any{"e": 1}}, true, `{"e":1}`},
			{"execution-denied", message.ExecutionDeniedOutput{Reason: "policy says no"}, false, "policy says no"},
			{"execution-denied-default", message.ExecutionDeniedOutput{}, false, "Tool call execution denied."},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				msg := message.Message{
					Role: message.RoleTool,
					Content: message.Content{Parts: []message.Part{
						message.ToolResultPart{ToolCallID: "id", ToolName: "t", Output: tc.output},
					}},
				}
				result := convertToolMessages(msg)
				block := result[0].Content[0]
				if block.IsError != tc.wantErr {
					t.Errorf("IsError = %v, want %v", block.IsError, tc.wantErr)
				}
				if got, ok := block.Content.(string); !ok || got != tc.wantWire {
					t.Errorf("Content = %v (%T), want %q", block.Content, block.Content, tc.wantWire)
				}
			})
		}
	})

	t.Run("content output with image produces anthropic image block", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleTool,
			Content: message.Content{Parts: []message.Part{
				message.ToolResultPart{
					ToolCallID: "id",
					ToolName:   "snap",
					Output: message.ContentOutput{Value: []message.ToolContentItem{
						{Type: "text", Text: "here it is"},
						{Type: "image-data", Data: "aW1n", MediaType: "image/png"},
					}},
				},
			}},
		}
		result := convertToolMessages(msg)
		block := result[0].Content[0]
		if block.IsError {
			t.Error("content output must not set is_error")
		}
		blocks, ok := block.Content.([]anthropicContentBlock)
		if !ok {
			t.Fatalf("expected []anthropicContentBlock, got %T", block.Content)
		}
		if len(blocks) != 2 {
			t.Fatalf("expected 2 blocks (text + image), got %d", len(blocks))
		}
		if blocks[0].Type != "text" || blocks[0].Text != "here it is" {
			t.Errorf("block[0] = %+v, want text 'here it is'", blocks[0])
		}
		if blocks[1].Type != "image" {
			t.Errorf("block[1].Type = %q, want image", blocks[1].Type)
		}
		if blocks[1].Source == nil || blocks[1].Source.MediaType != "image/png" || blocks[1].Source.Data != "aW1n" {
			t.Errorf("block[1].Source = %+v", blocks[1].Source)
		}
	})
}

// Tool conversion tests are in tools_test.go

func TestGetTextFromContent(t *testing.T) {
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

// TestConvertToAnthropicContent_URLSources verifies the url-source
// variants that ai-sdk emits for FilePart/ImagePart when the payload is
// remotely hosted rather than inlined (a FilePart with URL set, or an
// ImagePart whose Image is an http/https URL). Matches the switch in
// ai-sdk/packages/anthropic/src/convert-to-anthropic-messages-prompt.ts
// "case 'file'" / "case 'image'" url branches.
func TestConvertToAnthropicContent_URLSources(t *testing.T) {
	t.Run("ImagePart with http URL produces url source", func(t *testing.T) {
		content := message.Content{Parts: []message.Part{
			message.ImagePart{Image: "https://example.com/cat.png", MimeType: "image/png"},
		}}
		blocks := convertToAnthropicContent(content, nil)
		if len(blocks) != 1 || blocks[0].Type != "image" {
			t.Fatalf("unexpected blocks: %+v", blocks)
		}
		if blocks[0].Source.Type != "url" {
			t.Errorf("Source.Type = %q, want url", blocks[0].Source.Type)
		}
		if blocks[0].Source.URL != "https://example.com/cat.png" {
			t.Errorf("Source.URL = %q", blocks[0].Source.URL)
		}
		if blocks[0].Source.Data != "" {
			t.Errorf("Data should be empty for url source, got %q", blocks[0].Source.Data)
		}
	})

	t.Run("FilePart (pdf) with URL produces document/url source", func(t *testing.T) {
		content := message.Content{Parts: []message.Part{
			message.FilePart{URL: "https://example.com/doc.pdf", MimeType: "application/pdf", Filename: "doc.pdf"},
		}}
		blocks := convertToAnthropicContent(content, nil)
		if len(blocks) != 1 || blocks[0].Type != "document" {
			t.Fatalf("unexpected blocks: %+v", blocks)
		}
		if blocks[0].Source.Type != "url" || blocks[0].Source.URL != "https://example.com/doc.pdf" {
			t.Errorf("Source = %+v", blocks[0].Source)
		}
	})

	t.Run("FilePart (image/*) with URL produces image/url source", func(t *testing.T) {
		content := message.Content{Parts: []message.Part{
			message.FilePart{URL: "https://example.com/pic.jpg", MimeType: "image/jpeg"},
		}}
		blocks := convertToAnthropicContent(content, nil)
		if len(blocks) != 1 || blocks[0].Type != "image" {
			t.Fatalf("unexpected blocks: %+v", blocks)
		}
		if blocks[0].Source.Type != "url" {
			t.Errorf("Source.Type = %q, want url", blocks[0].Source.Type)
		}
	})

	t.Run("base64 FilePart still emits base64 source", func(t *testing.T) {
		content := message.Content{Parts: []message.Part{
			message.FilePart{Data: "base64data", MimeType: "application/pdf"},
		}}
		blocks := convertToAnthropicContent(content, nil)
		if blocks[0].Source.Type != "base64" {
			t.Errorf("base64 payload should keep base64 source, got %q", blocks[0].Source.Type)
		}
	})
}

// Tool-result messages may carry text + attachments (images/files)
// alongside ToolResultPart. Exercises convertToolMessages' attachment
// collection: ai-sdk ships the image inline after the tool_result text
// block in a single tool_result content array.
func TestConvertToolMessages_MultipartWithURLFile(t *testing.T) {
	msg := message.Message{
		Role: message.RoleTool,
		Content: message.Content{Parts: []message.Part{
			message.ToolResultPart{ToolCallID: "call-1", ToolName: "fetch", Output: message.TextOutput{Value: "ok"}},
			message.FilePart{URL: "https://example.com/report.pdf", MimeType: "application/pdf", Filename: "report.pdf"},
		}},
	}
	out := convertToolMessages(msg)
	if len(out) != 1 || len(out[0].Content) != 1 {
		t.Fatalf("unexpected message layout: %+v", out)
	}
	tr := out[0].Content[0]
	if tr.Type != "tool_result" || tr.ToolUseID != "call-1" {
		t.Fatalf("unexpected tool_result block: %+v", tr)
	}
	// Content should be a []anthropicContentBlock with text + document url-source.
	blocks, ok := tr.Content.([]anthropicContentBlock)
	if !ok {
		t.Fatalf("Content should be []anthropicContentBlock, got %T", tr.Content)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks (text + document), got %d", len(blocks))
	}
	// Marshal the second block and verify it's the url-source document.
	doc := blocks[1]
	if doc.Type != "document" {
		t.Errorf("block[1].Type = %q, want document", doc.Type)
	}
	if doc.Source.Type != "url" || doc.Source.URL != "https://example.com/report.pdf" {
		t.Errorf("block[1].Source = %+v", doc.Source)
	}
	// Wire shape — belt and suspenders: the url source must not emit
	// media_type/data fields.
	raw, _ := json.Marshal(doc.Source)
	if string(raw) != `{"type":"url","url":"https://example.com/report.pdf"}` {
		t.Errorf("url source wire shape = %s", raw)
	}
}
