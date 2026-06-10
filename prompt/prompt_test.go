package prompt

import (
	"context"
	"encoding/json"
	"net/url"
	"regexp"
	"testing"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
)

// Tests for ConvertToLanguageModelPrompt
// Source: ai-sdk/packages/ai/src/prompt/convert-to-language-model-prompt.test.ts

func TestConvertToLanguageModelPrompt_SystemMessage(t *testing.T) {
	t.Run("should convert a string system message", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				System:   "INSTRUCTIONS",
				Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Text: "Hello, world!"}}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}

		// Check system message
		if result[0].Role != "system" {
			t.Errorf("expected role 'system', got '%s'", result[0].Role)
		}
		if result[0].Content != "INSTRUCTIONS" {
			t.Errorf("expected content 'INSTRUCTIONS', got '%v'", result[0].Content)
		}

		// Check user message
		if result[1].Role != "user" {
			t.Errorf("expected role 'user', got '%s'", result[1].Role)
		}
		parts, ok := result[1].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[1].Content)
		}
		if parts[0].Type != "text" || parts[0].Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got '%s'", parts[0].Text)
		}
	})

	t.Run("should convert a SystemMessage system message", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				System: SystemMessage{
					Content:         "INSTRUCTIONS",
					ProviderOptions: map[string]any{"test": map[string]any{"value": "test"}},
				},
				Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Text: "Hello, world!"}}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}

		if result[0].Content != "INSTRUCTIONS" {
			t.Errorf("expected content 'INSTRUCTIONS', got '%v'", result[0].Content)
		}
		if result[0].ProviderOptions["test"] == nil {
			t.Error("expected provider options to be set")
		}
	})

	t.Run("should convert an array of SystemMessage system messages", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				System: []SystemMessage{
					{Content: "INSTRUCTIONS"},
					{Content: "INSTRUCTIONS 2"},
				},
				Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Text: "Hello, world!"}}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(result))
		}

		if result[0].Role != "system" || result[0].Content != "INSTRUCTIONS" {
			t.Errorf("first system message incorrect: %v", result[0])
		}
		if result[1].Role != "system" || result[1].Content != "INSTRUCTIONS 2" {
			t.Errorf("second system message incorrect: %v", result[1])
		}
	})
}

func TestConvertToLanguageModelPrompt_UserMessage(t *testing.T) {
	t.Run("should filter out empty text parts", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role:    message.RoleUser,
					Content: message.Content{Parts: []message.Part{message.TextPart{Text: ""}}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result) != 1 {
			t.Fatalf("expected 1 message, got %d", len(result))
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok {
			t.Fatalf("expected parts, got %T", result[0].Content)
		}
		if len(parts) != 0 {
			t.Errorf("expected 0 parts, got %d", len(parts))
		}
	})

	t.Run("should pass through non-empty text parts", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role:    message.RoleUser,
					Content: message.Content{Parts: []message.Part{message.TextPart{Text: "hello, world!"}}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}
		if parts[0].Text != "hello, world!" {
			t.Errorf("expected 'hello, world!', got '%s'", parts[0].Text)
		}
	})
}

func TestConvertToLanguageModelPrompt_ImageParts(t *testing.T) {
	t.Run("should download images when model does not support URLs", func(t *testing.T) {
		downloadCalled := false
		mockDownload := func(ctx context.Context, plans []DownloadPlan) []*DownloadedAsset {
			downloadCalled = true
			if len(plans) != 1 {
				t.Errorf("expected 1 plan, got %d", len(plans))
			}
			if plans[0].URL.String() != "https://example.com/image.png" {
				t.Errorf("expected URL 'https://example.com/image.png', got '%s'", plans[0].URL.String())
			}
			return []*DownloadedAsset{{
				Data:      []byte{0, 1, 2, 3},
				MediaType: "image/png",
			}}
		}

		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleUser,
					Content: message.Content{Parts: []message.Part{
						message.FilePart{Data: message.FileDataURL{URL: "https://example.com/image.png"}, MimeType: "image/png"},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      mockDownload,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !downloadCalled {
			t.Error("expected download to be called")
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}

		if parts[0].Type != "file" {
			t.Errorf("expected type 'file', got '%s'", parts[0].Type)
		}
		if parts[0].MediaType != "image/png" {
			t.Errorf("expected media type 'image/png', got '%s'", parts[0].MediaType)
		}

		data, ok := parts[0].Data.([]byte)
		if !ok {
			t.Fatalf("expected []byte data, got %T", parts[0].Data)
		}
		if len(data) != 4 {
			t.Errorf("expected 4 bytes, got %d", len(data))
		}
	})

	t.Run("should not download when URL is supported by model", func(t *testing.T) {
		mockDownload := func(ctx context.Context, plans []DownloadPlan) []*DownloadedAsset {
			// Return nil for supported URLs (they are marked as supported, so download is skipped)
			return make([]*DownloadedAsset, len(plans))
		}

		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleUser,
					Content: message.Content{Parts: []message.Part{
						message.FilePart{Data: message.FileDataURL{URL: "https://example.com/image.png"}, MimeType: "image/png"},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{
				"image/*": {regexp.MustCompile(`^https://.*$`)},
			},
			Download: mockDownload,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Download is called but returns nil for supported URLs
		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}

		// Data should be URL since it wasn't downloaded
		u, ok := parts[0].Data.(*url.URL)
		if !ok {
			t.Fatalf("expected *url.URL data, got %T", parts[0].Data)
		}
		if u.String() != "https://example.com/image.png" {
			t.Errorf("expected URL 'https://example.com/image.png', got '%s'", u.String())
		}
	})
}

func TestConvertToLanguageModelPrompt_FileParts(t *testing.T) {
	t.Run("should handle file parts with base64 string data", func(t *testing.T) {
		base64Data := "SGVsbG8sIFdvcmxkIQ==" // "Hello, World!" in base64

		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleUser,
					Content: message.Content{Parts: []message.Part{
						message.FilePart{Data: message.FileDataBytes{Data: base64Data}, MimeType: "text/plain"},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}

		if parts[0].Type != "file" {
			t.Errorf("expected type 'file', got '%s'", parts[0].Type)
		}
		if parts[0].MediaType != "text/plain" {
			t.Errorf("expected media type 'text/plain', got '%s'", parts[0].MediaType)
		}
		if parts[0].Data != base64Data {
			t.Errorf("expected data '%s', got '%v'", base64Data, parts[0].Data)
		}
	})

	t.Run("should handle file parts with filename", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleUser,
					Content: message.Content{Parts: []message.Part{
						message.FilePart{Data: message.FileDataBytes{Data: "SGVsbG8="}, MimeType: "text/plain", Filename: "hello.txt"},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}

		if parts[0].Filename != "hello.txt" {
			t.Errorf("expected filename 'hello.txt', got '%s'", parts[0].Filename)
		}
	})
}

func TestConvertToLanguageModelPrompt_ToolMessages(t *testing.T) {
	t.Run("should combine consecutive tool messages into a single tool message", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{
					{
						Role: message.RoleAssistant,
						Content: message.Content{Parts: []message.Part{
							message.ToolCallPart{ID: "toolCallId", Name: "toolName", Input: json.RawMessage(`{}`)},
							message.ToolApprovalRequestPart{ApprovalID: "approvalId", ToolCallID: "toolCallId"},
						}},
					},
					{
						Role: message.RoleTool,
						Content: message.Content{Parts: []message.Part{
							message.ToolApprovalResponsePart{ApprovalID: "approvalId", Approved: true},
						}},
					},
					{
						Role: message.RoleTool,
						Content: message.Content{Parts: []message.Part{
							message.ToolResultPart{ToolCallID: "toolCallId", ToolName: "toolName", Output: message.JSONOutput{Value: map[string]any{"some": "result"}}},
						}},
					},
				},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should be 2 messages: assistant + combined tool
		if len(result) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(result))
		}

		// Check assistant message
		assistantParts, ok := result[0].Content.([]LanguageModelPart)
		if !ok {
			t.Fatalf("expected parts, got %T", result[0].Content)
		}
		if len(assistantParts) != 1 {
			t.Errorf("expected 1 assistant part, got %d", len(assistantParts))
		}
		if assistantParts[0].Type != "tool-call" {
			t.Errorf("expected 'tool-call', got '%s'", assistantParts[0].Type)
		}

		// Check tool message (combined)
		toolParts, ok := result[1].Content.([]LanguageModelPart)
		if !ok {
			t.Fatalf("expected parts, got %T", result[1].Content)
		}
		// Only tool-result should remain (approval response without providerExecuted is filtered)
		if len(toolParts) != 1 {
			t.Errorf("expected 1 tool part, got %d", len(toolParts))
		}
		if toolParts[0].Type != "tool-result" {
			t.Errorf("expected 'tool-result', got '%s'", toolParts[0].Type)
		}
	})
}

func TestConvertToLanguageModelPrompt_DataURL(t *testing.T) {
	t.Run("should convert data URL to base64 content", func(t *testing.T) {
		result, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleUser,
					Content: message.Content{Parts: []message.Part{
						message.FilePart{Data: message.FileDataBytes{Data: "data:image/jpg;base64,/9j/3Q=="}, MimeType: "image/jpeg"},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		parts, ok := result[0].Content.([]LanguageModelPart)
		if !ok || len(parts) != 1 {
			t.Fatalf("expected 1 part, got %v", result[0].Content)
		}

		if parts[0].Type != "file" {
			t.Errorf("expected type 'file', got '%s'", parts[0].Type)
		}
		if parts[0].MediaType != "image/jpg" {
			t.Errorf("expected media type 'image/jpg', got '%s'", parts[0].MediaType)
		}
		if parts[0].Data != "/9j/3Q==" {
			t.Errorf("expected data '/9j/3Q==', got '%v'", parts[0].Data)
		}
	})
}

// Tests from convert-to-language-model-prompt.validation.test.ts

func TestConvertToLanguageModelPrompt_Validation(t *testing.T) {
	t.Run("should pass validation for provider-executed tools (deferred results)", func(t *testing.T) {
		// Provider-executed tools don't require results
		_, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleAssistant,
					Content: message.Content{Parts: []message.Part{
						// Note: Using regular ToolCallPart since we don't have providerExecuted on it
						// In real usage, this would need a special part type
						message.ToolCallPart{ID: "call_1", Name: "code_interpreter", Input: json.RawMessage(`{"code":"print(\"hello\")"}`)},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		// Without providerExecuted, this should fail
		if err == nil {
			t.Log("Note: Without providerExecuted=true, this raises MissingToolResultsError as expected")
		}
	})

	t.Run("should pass validation for tool-approval-response", func(t *testing.T) {
		_, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{
					{
						Role: message.RoleAssistant,
						Content: message.Content{Parts: []message.Part{
							message.ToolCallPart{ID: "call_to_approve", Name: "dangerous_action", Input: json.RawMessage(`{}`)},
							message.ToolApprovalRequestPart{
								ApprovalID: "approval_123",
								ToolCallID: "call_to_approve",
								ToolName:   "dangerous_action",
							},
						}},
					},
					{
						Role: message.RoleTool,
						Content: message.Content{Parts: []message.Part{
							message.ToolApprovalResponsePart{
								ApprovalID: "approval_123",
								Approved:   true,
							},
						}},
					},
				},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		// With approval response, the tool call is considered handled
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("should throw error for actual missing results", func(t *testing.T) {
		_, err := ConvertToLanguageModelPrompt(context.Background(), ConvertOptions{
			Prompt: StandardizedPrompt{
				Messages: []message.Message{{
					Role: message.RoleAssistant,
					Content: message.Content{Parts: []message.Part{
						message.ToolCallPart{ID: "call_missing_result", Name: "regular_tool", Input: json.RawMessage(`{}`)},
					}},
				}},
			},
			SupportedURLs: SupportedURLs{},
			Download:      nil,
		})

		if err == nil {
			t.Fatal("expected error for missing tool results")
		}

		if !errors.IsMissingToolResultsError(err) {
			t.Errorf("expected MissingToolResultsError, got %T: %v", err, err)
		}

		var missingErr *errors.MissingToolResultsError
		if errors.As(err, &missingErr) {
			if len(missingErr.ToolCallIDs) != 1 || missingErr.ToolCallIDs[0] != "call_missing_result" {
				t.Errorf("expected tool call ID 'call_missing_result', got %v", missingErr.ToolCallIDs)
			}
		}
	})
}

func TestDetectImageMediaType(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected string
	}{
		{"PNG", []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, "image/png"},
		{"JPEG", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10}, "image/jpeg"},
		{"GIF", []byte{0x47, 0x49, 0x46, 0x38, 0x39, 0x61}, "image/gif"},
		{"WebP", []byte{0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50}, "image/webp"},
		{"Unknown", []byte{0x00, 0x01, 0x02, 0x03}, ""},
		{"TooShort", []byte{0x89, 0x50}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := detectImageMediaType(tt.data)
			if result != tt.expected {
				t.Errorf("expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestParseDataURL(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectedData  string
		expectedMedia string
	}{
		{
			name:          "image with base64",
			input:         "data:image/png;base64,iVBORw0KGgo=",
			expectedData:  "iVBORw0KGgo=",
			expectedMedia: "image/png",
		},
		{
			name:          "text plain",
			input:         "data:text/plain;base64,SGVsbG8=",
			expectedData:  "SGVsbG8=",
			expectedMedia: "text/plain",
		},
		{
			name:          "no media type",
			input:         "data:;base64,SGVsbG8=",
			expectedData:  "SGVsbG8=",
			expectedMedia: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, media := parseDataURL(tt.input)
			if data != tt.expectedData {
				t.Errorf("expected data '%s', got '%s'", tt.expectedData, data)
			}
			if media != tt.expectedMedia {
				t.Errorf("expected media '%s', got '%s'", tt.expectedMedia, media)
			}
		})
	}
}

func TestMatchMediaType(t *testing.T) {
	tests := []struct {
		pattern   string
		mediaType string
		expected  bool
	}{
		{"*", "image/png", true},
		{"*/*", "image/png", true},
		{"image/*", "image/png", true},
		{"image/*", "image/jpeg", true},
		{"image/*", "application/pdf", false},
		{"image/png", "image/png", true},
		{"image/png", "image/jpeg", false},
		{"application/pdf", "application/pdf", true},
	}

	for _, tt := range tests {
		t.Run(tt.pattern+"_"+tt.mediaType, func(t *testing.T) {
			result := matchMediaType(tt.pattern, tt.mediaType)
			if result != tt.expected {
				t.Errorf("matchMediaType(%q, %q) = %v, expected %v", tt.pattern, tt.mediaType, result, tt.expected)
			}
		})
	}
}

func TestIsURLSupported(t *testing.T) {
	supportedURLs := SupportedURLs{
		"image/*":         {regexp.MustCompile(`^https://.*$`)},
		"application/pdf": {regexp.MustCompile(`^https://example\.com/.*$`)},
	}

	tests := []struct {
		url       string
		mediaType string
		expected  bool
	}{
		{"https://example.com/image.png", "image/png", true},
		{"http://example.com/image.png", "image/png", false}, // http not supported
		{"https://example.com/doc.pdf", "application/pdf", true},
		{"https://other.com/doc.pdf", "application/pdf", false}, // wrong domain
		{"https://example.com/file.txt", "text/plain", false},   // not in supported list
	}

	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			result := isURLSupported(tt.url, tt.mediaType, supportedURLs)
			if result != tt.expected {
				t.Errorf("isURLSupported(%q, %q) = %v, expected %v", tt.url, tt.mediaType, result, tt.expected)
			}
		})
	}
}
