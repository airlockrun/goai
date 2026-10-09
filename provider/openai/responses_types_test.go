package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
)

func convertToResponsesInput(messages []message.Message, systemMessageMode string, passThroughUnsupportedFiles bool) []responsesInputItem {
	result := convertToResponsesInputWithWarnings(messages, systemMessageMode, passThroughUnsupportedFiles)
	if result.Err != nil {
		panic(result.Err)
	}
	return result.Input
}

func TestConvertToResponsesInput_ContentOutput(t *testing.T) {
	detail := map[string]any{"openai": map[string]any{"imageDetail": "original"}}
	for _, tt := range []struct {
		name  string
		items []message.ToolContentItem
		want  string
	}{
		{"empty", nil, `[]`},
		{"empty text", []message.ToolContentItem{{Type: "text"}}, `[{"type":"input_text","text":""}]`},
		{"ordered mixed content", []message.ToolContentItem{
			{Type: "text", Text: "before"},
			{Type: "image-data", Data: "aW1n", MediaType: "image/png", ProviderOptions: detail},
			{Type: "text", Text: "between"},
			{Type: "file-data", Data: "cGRm", MediaType: "application/pdf", Filename: "catalogue.pdf"},
			{Type: "text", Text: "after"},
		}, `[{"type":"input_text","text":"before"},{"type":"input_image","image_url":"data:image/png;base64,aW1n","detail":"original"},{"type":"input_text","text":"between"},{"type":"input_file","file_data":"data:application/pdf;base64,cGRm","filename":"catalogue.pdf"},{"type":"input_text","text":"after"}]`},
		{"image URL", []message.ToolContentItem{{Type: "image-url", URL: "https://example.com/image.png", ProviderOptions: detail}}, `[{"type":"input_image","image_url":"https://example.com/image.png","detail":"original"}]`},
		{"image data URL", []message.ToolContentItem{{Type: "image-url", URL: "data:image/jpeg;base64,aW1n"}}, `[{"type":"input_image","image_url":"data:image/jpeg;base64,aW1n"}]`},
		{"file image data", []message.ToolContentItem{{Type: "file-data", Data: "aW1n", MediaType: "image/webp", ProviderOptions: detail}}, `[{"type":"input_image","image_url":"data:image/webp;base64,aW1n","detail":"original"}]`},
		{"file image URL", []message.ToolContentItem{{Type: "file-url", URL: "https://example.com/image.gif", MediaType: "image/gif", ProviderOptions: detail}}, `[{"type":"input_image","image_url":"https://example.com/image.gif","detail":"original"}]`},
		{"file URL", []message.ToolContentItem{{Type: "file-url", URL: "https://example.com/document.pdf", MediaType: "application/pdf"}}, `[{"type":"input_file","file_url":"https://example.com/document.pdf"}]`},
		{"file data URL", []message.ToolContentItem{{Type: "file-url", URL: "data:application/pdf;base64,cGRm", MediaType: "application/pdf"}}, `[{"type":"input_file","file_url":"data:application/pdf;base64,cGRm"}]`},
		{"file data without filename", []message.ToolContentItem{{Type: "file-data", Data: "cGRm", MediaType: "application/pdf"}}, `[{"type":"input_file","file_data":"data:application/pdf;base64,cGRm","filename":"data"}]`},
		{"other file media", []message.ToolContentItem{{Type: "file-data", Data: "dGV4dA==", MediaType: "text/plain", Filename: "note.txt"}}, `[{"type":"input_file","file_data":"data:text/plain;base64,dGV4dA==","filename":"note.txt"}]`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			converted := convertToResponsesInputWithWarnings([]message.Message{
				message.NewToolMessage("call-image", "read", message.ContentOutput{Value: tt.items}),
			}, "system", false)
			if converted.Err != nil || len(converted.Warnings) != 0 || len(converted.Input) != 1 {
				t.Fatalf("conversion = %+v", converted)
			}
			item := converted.Input[0]
			if item.CallID != "call-image" || item.Type != "function_call_output" {
				t.Fatalf("tool association = %+v", item)
			}
			data, err := json.Marshal(item.Output)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tt.want), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("output = %s, want %s", data, tt.want)
			}
		})
	}
}

func TestConvertToResponsesInput_ContentOutputAssociation(t *testing.T) {
	converted := convertToResponsesInputWithWarnings([]message.Message{{Role: message.RoleTool, Content: message.Content{Parts: []message.Part{
		message.ToolResultPart{ToolCallID: "first", ToolName: "read", Output: message.ContentOutput{Value: []message.ToolContentItem{
			{Type: "text", Text: "first image"}, {Type: "image-data", Data: "Zmlyc3Q=", MediaType: "image/png"},
		}}},
		message.FilePart{Data: message.FileDataURL{URL: "https://example.com/sibling.png"}, MimeType: "image/png"},
		message.ToolResultPart{ToolCallID: "second", ToolName: "read", Output: message.ContentOutput{Value: []message.ToolContentItem{
			{Type: "image-url", URL: "https://example.com/second.png"}, {Type: "text", Text: "second image"},
		}}},
	}}}}, "system", false)
	if converted.Err != nil || len(converted.Input) != 2 {
		t.Fatalf("conversion = %+v", converted)
	}
	first := converted.Input[0]
	second := converted.Input[1]
	if first.CallID != "first" || second.CallID != "second" {
		t.Fatalf("call IDs = %q, %q", first.CallID, second.CallID)
	}
	wantFirst := []responsesContentPart{{Type: "input_text", Text: "first image"}, {Type: "input_image", ImageURL: "data:image/png;base64,Zmlyc3Q="}}
	wantSecond := []responsesContentPart{{Type: "input_image", ImageURL: "https://example.com/second.png"}, {Type: "input_text", Text: "second image"}, {Type: "input_image", ImageURL: "https://example.com/sibling.png"}}
	if !reflect.DeepEqual(first.Output, wantFirst) || !reflect.DeepEqual(second.Output, wantSecond) {
		t.Fatalf("outputs = %#v, %#v", first.Output, second.Output)
	}
}

func TestConvertToResponsesInput_ToolAttachmentErrors(t *testing.T) {
	for _, tt := range []struct {
		name  string
		parts []message.Part
	}{
		{"missing result", []message.Part{message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "aW1n"}}}},
		{"missing media type", []message.Part{message.ToolResultPart{ToolCallID: "call", Output: message.TextOutput{}}, message.FilePart{Data: message.FileDataBytes{Data: "aW1n"}}}},
		{"reference", []message.Part{message.ToolResultPart{ToolCallID: "call", Output: message.TextOutput{}}, message.FilePart{Data: message.FileDataReference{Reference: map[string]any{"openai": "file-1"}}}}},
		{"text file data", []message.Part{message.ToolResultPart{ToolCallID: "call", Output: message.TextOutput{}}, message.FilePart{Data: message.FileDataText{Text: "text"}}}},
		{"unassociated text", []message.Part{message.ToolResultPart{ToolCallID: "call", Output: message.TextOutput{}}, message.TextPart{Text: "text"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToResponsesInputWithWarnings([]message.Message{{Role: message.RoleTool, Content: message.Content{Parts: tt.parts}}}, "system", false)
			if result.Err == nil || len(result.Input) != 0 {
				t.Fatalf("conversion = %+v", result)
			}
		})
	}
}

func TestConvertToResponsesInput_ScalarToolOutputs(t *testing.T) {
	for _, tt := range []struct {
		name   string
		output message.ToolResultOutput
		want   string
	}{
		{"text", message.TextOutput{Value: "ok"}, "ok"},
		{"empty", message.TextOutput{}, ""},
		{"JSON", message.JSONOutput{Value: map[string]any{"ok": true}}, `{"ok":true}`},
		{"error", message.ErrorTextOutput{Value: "failed"}, "failed"},
		{"error JSON", message.ErrorJSONOutput{Value: map[string]any{"failed": true}}, `{"failed":true}`},
		{"denied", message.ExecutionDeniedOutput{Reason: "denied"}, "denied"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToResponsesInputWithWarnings([]message.Message{message.NewToolMessage("call", "read", tt.output)}, "system", false)
			if result.Err != nil || len(result.Input) != 1 || result.Input[0].Output != tt.want {
				t.Fatalf("conversion = %+v, want scalar %q", result, tt.want)
			}
		})
	}
}

// Tests for convertToResponsesInput - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/responses/convert-to-openai-responses-input.test.ts

func TestConvertToResponsesInput_SystemMessages(t *testing.T) {
	t.Run("should convert system messages to system role", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Role != "system" {
			t.Errorf("expected role 'system', got '%s'", result[0].Role)
		}
		if result[0].Content != "Hello" {
			t.Errorf("expected content 'Hello', got '%s'", result[0].Content)
		}
	})

	t.Run("should convert system messages to developer role", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "developer", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Role != "developer" {
			t.Errorf("expected role 'developer', got '%s'", result[0].Role)
		}
		if result[0].Content != "Hello" {
			t.Errorf("expected content 'Hello', got '%s'", result[0].Content)
		}
	})

	t.Run("should remove system messages when mode is remove", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "remove", false)

		if len(result) != 0 {
			t.Fatalf("expected 0 items, got %d", len(result))
		}
	})

	t.Run("should default to developer for unknown mode", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "unknown", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Role != "developer" {
			t.Errorf("expected role 'developer', got '%s'", result[0].Role)
		}
	})
}

func TestConvertToResponsesInput_UserMessages(t *testing.T) {
	t.Run("should convert user message with text", func(t *testing.T) {
		messages := []message.Message{
			message.NewUserMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Role != "user" {
			t.Errorf("expected role 'user', got '%s'", result[0].Role)
		}
		if len(result[0].ContentParts) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(result[0].ContentParts))
		}
		if result[0].ContentParts[0].Type != "input_text" {
			t.Errorf("expected type 'input_text', got '%s'", result[0].ContentParts[0].Type)
		}
		if result[0].ContentParts[0].Text != "Hello" {
			t.Errorf("expected text 'Hello', got '%s'", result[0].ContentParts[0].Text)
		}
	})

	t.Run("should convert user message with image URL", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleUser,
				Content: message.Content{
					Parts: []message.Part{
						message.TextPart{Text: "Hello"},
						message.FilePart{Data: message.FileDataURL{URL: "https://example.com/image.jpg"}, MimeType: "image/jpeg"},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if len(result[0].ContentParts) != 2 {
			t.Fatalf("expected 2 content parts, got %d", len(result[0].ContentParts))
		}
		if result[0].ContentParts[1].Type != "input_image" {
			t.Errorf("expected type 'input_image', got '%s'", result[0].ContentParts[1].Type)
		}
		if result[0].ContentParts[1].ImageURL != "https://example.com/image.jpg" {
			t.Errorf("expected image URL 'https://example.com/image.jpg', got '%s'", result[0].ContentParts[1].ImageURL)
		}
	})
}

func TestConvertToResponsesInput_AssistantMessages(t *testing.T) {
	t.Run("should convert assistant message with text", func(t *testing.T) {
		messages := []message.Message{
			message.NewAssistantMessage("Hello"),
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Role != "assistant" {
			t.Errorf("expected role 'assistant', got '%s'", result[0].Role)
		}
		if len(result[0].AssistantContent) != 1 {
			t.Fatalf("expected 1 content item, got %d", len(result[0].AssistantContent))
		}
		if result[0].AssistantContent[0].Type != "output_text" {
			t.Errorf("expected type 'output_text', got '%s'", result[0].AssistantContent[0].Type)
		}
		if result[0].AssistantContent[0].Text != "Hello" {
			t.Errorf("expected text 'Hello', got '%s'", result[0].AssistantContent[0].Text)
		}
	})

	t.Run("should convert assistant message with tool calls", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolCallPart{
							ID:    "call_123",
							Name:  "get_weather",
							Input: json.RawMessage(`{"location":"NYC"}`),
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		// Should have function_call item
		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Type != "function_call" {
			t.Errorf("expected type 'function_call', got '%s'", result[0].Type)
		}
		if result[0].CallID != "call_123" {
			t.Errorf("expected call_id 'call_123', got '%s'", result[0].CallID)
		}
		if result[0].Name != "get_weather" {
			t.Errorf("expected name 'get_weather', got '%s'", result[0].Name)
		}
	})
}

func TestConvertToResponsesInput_ToolMessages(t *testing.T) {
	t.Run("should convert tool result message", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "get_weather",
							Output:     message.TextOutput{Value: "Sunny, 72°F"},
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Type != "function_call_output" {
			t.Errorf("expected type 'function_call_output', got '%s'", result[0].Type)
		}
		if result[0].CallID != "call_123" {
			t.Errorf("expected call_id 'call_123', got '%s'", result[0].CallID)
		}
		if result[0].Output != "Sunny, 72°F" {
			t.Errorf("expected output 'Sunny, 72°F', got '%v'", result[0].Output)
		}
	})

	t.Run("should convert tool result with JSON object", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "get_weather",
							Output:     message.JSONOutput{Value: map[string]any{"temp": 72, "condition": "sunny"}},
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		// Should be JSON stringified
		outputStr, ok := result[0].Output.(string)
		if !ok {
			t.Fatalf("expected string output, got %T", result[0].Output)
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(outputStr), &parsed); err != nil {
			t.Fatalf("expected valid JSON output, got error: %v", err)
		}
		if parsed["temp"].(float64) != 72 {
			t.Errorf("expected temp 72, got %v", parsed["temp"])
		}
	})
}

// Tests for tool result multipart content — translated from ai-sdk
// Source: ai-sdk/packages/openai/src/responses/convert-to-openai-responses-input.test.ts

func TestConvertToResponsesInput_ToolResultMultipart(t *testing.T) {
	t.Run("should convert tool result with image data", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "search",
							Output:     message.TextOutput{Value: ""},
						},
						message.FilePart{
							Data:     message.FileDataBytes{Data: "base64_data"},
							MimeType: "image/png",
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		if len(result) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result))
		}
		if result[0].Type != "function_call_output" {
			t.Errorf("expected type 'function_call_output', got '%s'", result[0].Type)
		}
		if result[0].CallID != "call_123" {
			t.Errorf("expected call_id 'call_123', got '%s'", result[0].CallID)
		}
		// Output should be array of content parts
		parts, ok := result[0].Output.([]responsesContentPart)
		if !ok {
			t.Fatalf("expected []responsesContentPart output, got %T", result[0].Output)
		}
		if len(parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(parts))
		}
		if parts[0].Type != "input_image" {
			t.Errorf("expected type 'input_image', got '%s'", parts[0].Type)
		}
		if parts[0].ImageURL != "data:image/png;base64,base64_data" {
			t.Errorf("expected data URI, got '%s'", parts[0].ImageURL)
		}
	})

	t.Run("should convert tool result with image URL", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "screenshot",
							Output:     message.TextOutput{Value: ""},
						},
						message.FilePart{
							Data:     message.FileDataURL{URL: "https://example.com/screenshot.png"},
							MimeType: "image/png",
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		parts, ok := result[0].Output.([]responsesContentPart)
		if !ok {
			t.Fatalf("expected []responsesContentPart output, got %T", result[0].Output)
		}
		if parts[0].ImageURL != "https://example.com/screenshot.png" {
			t.Errorf("expected URL preserved, got '%s'", parts[0].ImageURL)
		}
	})

	t.Run("should convert tool result with PDF file", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "search",
							Output:     message.TextOutput{Value: ""},
						},
						message.FilePart{
							Data:     message.FileDataBytes{Data: "AQIDBAU="},
							MimeType: "application/pdf",
							Filename: "document.pdf",
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		parts, ok := result[0].Output.([]responsesContentPart)
		if !ok {
			t.Fatalf("expected []responsesContentPart output, got %T", result[0].Output)
		}
		if parts[0].Type != "input_file" {
			t.Errorf("expected type 'input_file', got '%s'", parts[0].Type)
		}
		if parts[0].FileData != "data:application/pdf;base64,AQIDBAU=" {
			t.Errorf("expected data URI, got '%s'", parts[0].FileData)
		}
		if parts[0].Filename != "document.pdf" {
			t.Errorf("expected filename 'document.pdf', got '%s'", parts[0].Filename)
		}
	})

	// ai-sdk #bc01093: URL-bearing FilePart in tool output maps to
	// input_file with file_url (no file_data, no PDF restriction).
	t.Run("should convert tool result with file URL", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_789",
							ToolName:   "fetch_report",
							Output:     message.TextOutput{Value: ""},
						},
						message.FilePart{
							Data:     message.FileDataURL{URL: "https://files.example.com/q4.pdf"},
							MimeType: "application/pdf",
							Filename: "q4.pdf",
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		parts, ok := result[0].Output.([]responsesContentPart)
		if !ok {
			t.Fatalf("expected []responsesContentPart output, got %T", result[0].Output)
		}
		if parts[0].Type != "input_file" {
			t.Errorf("expected type 'input_file', got %q", parts[0].Type)
		}
		if parts[0].FileURL != "https://files.example.com/q4.pdf" {
			t.Errorf("expected FileURL preserved, got %q", parts[0].FileURL)
		}
		if parts[0].Filename != "q4.pdf" {
			t.Errorf("expected filename q4.pdf, got %q", parts[0].Filename)
		}
		if parts[0].FileData != "" {
			t.Errorf("FileData should be empty for URL-based file, got %q", parts[0].FileData)
		}
	})

	t.Run("should convert tool result with mixed content", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_123",
							ToolName:   "search",
							Output:     message.TextOutput{Value: "The weather in San Francisco is 72°F"},
						},
						message.FilePart{
							Data:     message.FileDataBytes{Data: "base64_data"},
							MimeType: "image/png",
						},
						message.FilePart{
							Data:     message.FileDataBytes{Data: "AQIDBAU="},
							MimeType: "application/pdf",
						},
					},
				},
			},
		}

		result := convertToResponsesInput(messages, "system", false)

		parts, ok := result[0].Output.([]responsesContentPart)
		if !ok {
			t.Fatalf("expected []responsesContentPart output, got %T", result[0].Output)
		}
		if len(parts) != 3 {
			t.Fatalf("expected 3 parts (text + image + file), got %d", len(parts))
		}
		if parts[0].Type != "input_text" || parts[0].Text != "The weather in San Francisco is 72°F" {
			t.Errorf("expected input_text with weather, got %+v", parts[0])
		}
		if parts[1].Type != "input_image" || parts[1].ImageURL != "data:image/png;base64,base64_data" {
			t.Errorf("expected input_image, got %+v", parts[1])
		}
		if parts[2].Type != "input_file" || parts[2].FileData != "data:application/pdf;base64,AQIDBAU=" {
			t.Errorf("expected input_file, got %+v", parts[2])
		}
	})
}

func TestConvertToResponsesInput_MixedConversation(t *testing.T) {
	t.Run("should handle full conversation flow", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("You are a helpful assistant"),
			message.NewUserMessage("What's the weather?"),
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolCallPart{
							ID:    "call_1",
							Name:  "get_weather",
							Input: json.RawMessage(`{"location":"NYC"}`),
						},
					},
				},
			},
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{
							ToolCallID: "call_1",
							ToolName:   "get_weather",
							Output:     message.TextOutput{Value: "Sunny"},
						},
					},
				},
			},
			message.NewAssistantMessage("The weather is sunny!"),
		}

		result := convertToResponsesInput(messages, "system", false)

		// Should have: system, user, function_call, function_call_output, assistant
		if len(result) != 5 {
			t.Fatalf("expected 5 items, got %d", len(result))
		}

		if result[0].Role != "system" {
			t.Errorf("item 0: expected role 'system', got '%s'", result[0].Role)
		}
		if result[1].Role != "user" {
			t.Errorf("item 1: expected role 'user', got '%s'", result[1].Role)
		}
		if result[2].Type != "function_call" {
			t.Errorf("item 2: expected type 'function_call', got '%s'", result[2].Type)
		}
		if result[3].Type != "function_call_output" {
			t.Errorf("item 3: expected type 'function_call_output', got '%s'", result[3].Type)
		}
		if result[4].Role != "assistant" {
			t.Errorf("item 4: expected role 'assistant', got '%s'", result[4].Role)
		}
	})
}

func TestResponsesInputItem_MarshalJSON(t *testing.T) {
	t.Run("should marshal system message correctly", func(t *testing.T) {
		item := responsesInputItem{
			Role:    "system",
			Content: "Hello",
		}

		data, err := json.Marshal(item)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["role"] != "system" {
			t.Errorf("expected role 'system', got '%v'", result["role"])
		}
		if result["content"] != "Hello" {
			t.Errorf("expected content 'Hello', got '%v'", result["content"])
		}
	})

	t.Run("should marshal developer message correctly", func(t *testing.T) {
		item := responsesInputItem{
			Role:    "developer",
			Content: "Instructions",
		}

		data, err := json.Marshal(item)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["role"] != "developer" {
			t.Errorf("expected role 'developer', got '%v'", result["role"])
		}
	})

	t.Run("should marshal function_call correctly", func(t *testing.T) {
		item := responsesInputItem{
			Type:      "function_call",
			CallID:    "call_123",
			Name:      "get_weather",
			Arguments: `{"location":"NYC"}`,
		}

		data, err := json.Marshal(item)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["type"] != "function_call" {
			t.Errorf("expected type 'function_call', got '%v'", result["type"])
		}
		if result["call_id"] != "call_123" {
			t.Errorf("expected call_id 'call_123', got '%v'", result["call_id"])
		}
	})

	t.Run("should marshal reasoning correctly", func(t *testing.T) {
		item := responsesInputItem{
			Type:             "reasoning",
			ID:               "reasoning_001",
			EncryptedContent: "gAAAA...",
			Summary: []responsesSummaryPart{
				{Type: "summary_text", Text: "First reasoning step"},
				{Type: "summary_text", Text: "Second reasoning step"},
			},
		}

		data, err := json.Marshal(item)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["type"] != "reasoning" {
			t.Errorf("expected type 'reasoning', got '%v'", result["type"])
		}
		if result["id"] != "reasoning_001" {
			t.Errorf("expected id 'reasoning_001', got '%v'", result["id"])
		}
		if result["encrypted_content"] != "gAAAA..." {
			t.Errorf("expected encrypted_content 'gAAAA...', got '%v'", result["encrypted_content"])
		}
		summary := result["summary"].([]any)
		if len(summary) != 2 {
			t.Errorf("expected 2 summary parts, got %d", len(summary))
		}
	})
}

// Tests for reasoning conversion - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/responses/convert-to-openai-responses-input.test.ts

func TestConvertToResponsesInput_ReasoningParts(t *testing.T) {
	t.Run("should convert reasoning with encrypted content", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "Thinking about the problem...",
							ProviderOptions: map[string]any{
								"itemId":                    "reasoning_001",
								"reasoningEncryptedContent": "gAAAA...",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		if len(result.Input) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result.Input))
		}
		if result.Input[0].Type != "reasoning" {
			t.Errorf("expected type 'reasoning', got '%s'", result.Input[0].Type)
		}
		if result.Input[0].ID != "reasoning_001" {
			t.Errorf("expected id 'reasoning_001', got '%s'", result.Input[0].ID)
		}
		if result.Input[0].EncryptedContent != "gAAAA..." {
			t.Errorf("expected encrypted_content 'gAAAA...', got '%s'", result.Input[0].EncryptedContent)
		}
		if len(result.Input[0].Summary) != 1 {
			t.Fatalf("expected 1 summary part, got %d", len(result.Input[0].Summary))
		}
		if result.Input[0].Summary[0].Text != "Thinking about the problem..." {
			t.Errorf("expected summary text 'Thinking about the problem...', got '%s'", result.Input[0].Summary[0].Text)
		}
		if len(result.Warnings) != 0 {
			t.Errorf("expected no warnings, got %d", len(result.Warnings))
		}
	})

	t.Run("should merge consecutive parts with same reasoning ID", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "First reasoning step",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
						message.ReasoningPart{
							Text: "Second reasoning step",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		// Should be merged into single reasoning message
		if len(result.Input) != 1 {
			t.Fatalf("expected 1 item (merged), got %d", len(result.Input))
		}
		if result.Input[0].Type != "reasoning" {
			t.Errorf("expected type 'reasoning', got '%s'", result.Input[0].Type)
		}
		if len(result.Input[0].Summary) != 2 {
			t.Fatalf("expected 2 summary parts, got %d", len(result.Input[0].Summary))
		}
		if result.Input[0].Summary[0].Text != "First reasoning step" {
			t.Errorf("expected first summary 'First reasoning step', got '%s'", result.Input[0].Summary[0].Text)
		}
		if result.Input[0].Summary[1].Text != "Second reasoning step" {
			t.Errorf("expected second summary 'Second reasoning step', got '%s'", result.Input[0].Summary[1].Text)
		}
		if len(result.Warnings) != 0 {
			t.Errorf("expected no warnings, got %d", len(result.Warnings))
		}
	})

	t.Run("should create separate messages for different reasoning IDs", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "First reasoning block",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
						message.ReasoningPart{
							Text: "Second reasoning block",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_002",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		// Should have separate reasoning messages
		if len(result.Input) != 2 {
			t.Fatalf("expected 2 items, got %d", len(result.Input))
		}
		if result.Input[0].ID != "reasoning_001" {
			t.Errorf("expected first id 'reasoning_001', got '%s'", result.Input[0].ID)
		}
		if result.Input[1].ID != "reasoning_002" {
			t.Errorf("expected second id 'reasoning_002', got '%s'", result.Input[1].ID)
		}
		if len(result.Warnings) != 0 {
			t.Errorf("expected no warnings, got %d", len(result.Warnings))
		}
	})

	t.Run("should warn when reasoning part lacks itemId AND encrypted content", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "This is a reasoning part without provider options",
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		// Should skip the reasoning part and add a warning
		if len(result.Input) != 0 {
			t.Fatalf("expected 0 items (skipped), got %d", len(result.Input))
		}
		if len(result.Warnings) != 1 {
			t.Fatalf("expected 1 warning, got %d", len(result.Warnings))
		}
		if result.Warnings[0].Type != "other" {
			t.Errorf("expected warning type 'other', got '%s'", result.Warnings[0].Type)
		}
	})

	// ai-sdk #5e18272: a reasoning part WITHOUT itemId but WITH
	// encrypted_content should round-trip as a reasoning item (with no
	// id) so multi-turn reasoning works in ZDR / store:false mode.
	t.Run("should round-trip reasoning without itemId when encrypted content is present", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "Thinking step",
							ProviderOptions: map[string]any{
								"reasoningEncryptedContent": "enc_abc123",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		if len(result.Warnings) != 0 {
			t.Errorf("expected 0 warnings, got %d: %+v", len(result.Warnings), result.Warnings)
		}
		if len(result.Input) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result.Input))
		}
		item := result.Input[0]
		if item.Type != "reasoning" {
			t.Errorf("Type = %q, want reasoning", item.Type)
		}
		if item.ID != "" {
			t.Errorf("ID = %q, want empty (itemId missing)", item.ID)
		}
		if item.EncryptedContent != "enc_abc123" {
			t.Errorf("EncryptedContent = %q, want enc_abc123", item.EncryptedContent)
		}
		if len(item.Summary) != 1 || item.Summary[0].Text != "Thinking step" {
			t.Errorf("Summary = %+v", item.Summary)
		}
	})

	t.Run("should warn when appending empty reasoning part to existing sequence", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "First reasoning step",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
						message.ReasoningPart{
							Text: "", // Empty text
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		// Should have one reasoning message
		if len(result.Input) != 1 {
			t.Fatalf("expected 1 item, got %d", len(result.Input))
		}
		// Should have one summary part (the non-empty one)
		if len(result.Input[0].Summary) != 1 {
			t.Errorf("expected 1 summary part, got %d", len(result.Input[0].Summary))
		}
		// Should have a warning about empty reasoning
		if len(result.Warnings) != 1 {
			t.Fatalf("expected 1 warning, got %d", len(result.Warnings))
		}
	})

	t.Run("should update encrypted content from later part", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "First reasoning step",
							ProviderOptions: map[string]any{
								"itemId": "reasoning_001",
							},
						},
						message.ReasoningPart{
							Text: "Second reasoning step",
							ProviderOptions: map[string]any{
								"itemId":                    "reasoning_001",
								"reasoningEncryptedContent": "gAAAA-updated",
							},
						},
					},
				},
			},
		}

		result := convertToResponsesInputWithWarnings(messages, "system", false)

		// Should have encrypted content from the second part
		if result.Input[0].EncryptedContent != "gAAAA-updated" {
			t.Errorf("expected encrypted_content 'gAAAA-updated', got '%s'", result.Input[0].EncryptedContent)
		}
	})

	t.Run("should warn when removing system messages", func(t *testing.T) {
		messages := []message.Message{
			message.NewSystemMessage("Hello"),
		}

		result := convertToResponsesInputWithWarnings(messages, "remove", false)

		if len(result.Input) != 0 {
			t.Errorf("expected 0 items, got %d", len(result.Input))
		}
		if len(result.Warnings) != 1 {
			t.Fatalf("expected 1 warning, got %d", len(result.Warnings))
		}
		if result.Warnings[0].Message != "system messages are removed for this model" {
			t.Errorf("expected specific warning message, got '%s'", result.Warnings[0].Message)
		}
	})
}

// Regression for ai-sdk #66a374c: phase on an assistant text part,
// forwarded via providerOptions.openai.phase, echoes back on the
// follow-up request so gpt-5.3-codex doesn't early-stop.
func TestConvertToResponsesInput_PhaseEchoesBack(t *testing.T) {
	msgs := []message.Message{
		{
			Role: message.RoleAssistant,
			Content: message.Content{
				Parts: []message.Part{
					message.TextPart{
						Text: "Working on it",
						ProviderOptions: map[string]any{
							"openai": map[string]any{"phase": "commentary"},
						},
					},
				},
			},
		},
	}
	items := convertToResponsesInput(msgs, "system", false)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	if items[0].Phase != "commentary" {
		t.Errorf("Phase = %q, want commentary", items[0].Phase)
	}
	raw, err := json.Marshal(items[0])
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	_ = json.Unmarshal(raw, &decoded)
	if decoded["phase"] != "commentary" {
		t.Errorf("marshaled JSON missing phase=commentary, got %s", raw)
	}
}

// Namespace round-trips on function_call input items (ai-sdk #15193). When a
// tool-call carries providerMetadata.openai.namespace, the serialized
// function_call must echo it or OpenAI rejects the follow-up.
func TestConvertToResponsesInput_NamespaceRoundTrip(t *testing.T) {
	t.Run("forwards namespace from providerOptions.openai.namespace", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolCallPart{
							ID:    "call_1",
							Name:  "search",
							Input: json.RawMessage(`{"q":"x"}`),
							ProviderOptions: map[string]any{
								"openai": map[string]any{"namespace": "mcp_server"},
							},
						},
					},
				},
			},
		}
		result := convertToResponsesInput(messages, "system", false)
		if len(result) != 1 || result[0].Type != "function_call" {
			t.Fatalf("expected one function_call item, got %+v", result)
		}
		if result[0].Namespace != "mcp_server" {
			t.Errorf("namespace = %q, want mcp_server", result[0].Namespace)
		}
		b, err := json.Marshal(result[0])
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		if m["namespace"] != "mcp_server" {
			t.Errorf("serialized namespace = %v, want mcp_server", m["namespace"])
		}
	})

	t.Run("omits namespace when absent", func(t *testing.T) {
		messages := []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolCallPart{ID: "call_1", Name: "search", Input: json.RawMessage(`{}`)},
					},
				},
			},
		}
		b, err := json.Marshal(convertToResponsesInput(messages, "system", false)[0])
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		if _, ok := m["namespace"]; ok {
			t.Errorf("namespace should be omitted, got %v", m["namespace"])
		}
	})
}

// Opt-in pass-through for unsupported file media types (ai-sdk #15297).
func TestConvertToResponsesContentParts_PassThroughUnsupportedFiles(t *testing.T) {
	content := message.Content{
		Parts: []message.Part{
			message.FilePart{MimeType: "text/csv", Data: message.FileDataBytes{Data: "Zm9v"}, Filename: "data.csv"},
		},
	}

	t.Run("drops non-image non-pdf files by default", func(t *testing.T) {
		parts := convertToResponsesContentParts(content, false)
		if len(parts) != 0 {
			t.Errorf("expected file dropped, got %+v", parts)
		}
	})

	t.Run("forwards file with its media type when enabled", func(t *testing.T) {
		parts := convertToResponsesContentParts(content, true)
		if len(parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(parts))
		}
		if parts[0].Type != "input_file" {
			t.Errorf("type = %q, want input_file", parts[0].Type)
		}
		if parts[0].Filename != "data.csv" {
			t.Errorf("filename = %q, want data.csv", parts[0].Filename)
		}
		want := "data:text/csv;base64,Zm9v"
		if parts[0].FileData != want {
			t.Errorf("fileData = %q, want %q", parts[0].FileData, want)
		}
	})
}
