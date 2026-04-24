package openai

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for convertToChatTools - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/chat/openai-chat-prepare-tools.test.ts
//
// Note: ai-sdk's prepareChatTools returns:
// - tools: []ChatTool
// - toolChoice: 'auto' | 'none' | 'required' | {type: 'function', function: {name: string}}
// - toolWarnings: []Warning
//
// goai's convertToChatTools only returns []chatTool, no toolChoice or warnings.
// toolChoice handling is done separately in the request building.

func TestConvertToChatTools_FunctionTools(t *testing.T) {
	t.Run("should return empty array when no tools", func(t *testing.T) {
		tools := tool.Set{}

		result := convertToChatTools(tools.Ordered(nil))

		if len(result) != 0 {
			t.Fatalf("expected 0 tools, got %d", len(result))
		}
	})

	t.Run("should correctly prepare function tools", func(t *testing.T) {
		tools := tool.Set{
			"testFunction": {
				Name:        "testFunction",
				Description: "A test function",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
		}

		result := convertToChatTools(tools.Ordered(nil))

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}

		// ai-sdk chat tools have structure: {type: "function", function: {...}}
		if result[0].Type != "function" {
			t.Errorf("expected type 'function', got '%s'", result[0].Type)
		}
		if result[0].Function.Name != "testFunction" {
			t.Errorf("expected function name 'testFunction', got '%s'", result[0].Function.Name)
		}
		if result[0].Function.Description != "A test function" {
			t.Errorf("expected description 'A test function', got '%s'", result[0].Function.Description)
		}
	})

	t.Run("should handle multiple tools", func(t *testing.T) {
		tools := tool.Set{
			"tool1": {
				Name:        "tool1",
				Description: "First tool",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
			"tool2": {
				Name:        "tool2",
				Description: "Second tool",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {"input": {"type": "string"}}}`),
			},
		}

		result := convertToChatTools(tools.Ordered(nil))

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		// Find each tool by name
		var foundTool1, foundTool2 bool
		for _, tool := range result {
			if tool.Function.Name == "tool1" {
				foundTool1 = true
				if tool.Function.Description != "First tool" {
					t.Errorf("tool1: expected description 'First tool', got '%s'", tool.Function.Description)
				}
			}
			if tool.Function.Name == "tool2" {
				foundTool2 = true
				if tool.Function.Description != "Second tool" {
					t.Errorf("tool2: expected description 'Second tool', got '%s'", tool.Function.Description)
				}
			}
		}

		if !foundTool1 {
			t.Error("expected to find tool1")
		}
		if !foundTool2 {
			t.Error("expected to find tool2")
		}
	})

	t.Run("should filter tools by activeTools", func(t *testing.T) {
		tools := tool.Set{
			"tool1": {
				Name:        "tool1",
				Description: "First tool",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
			"tool2": {
				Name:        "tool2",
				Description: "Second tool",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
			"tool3": {
				Name:        "tool3",
				Description: "Third tool",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
		}

		result := convertToChatTools(tools.Ordered([]string{"tool1", "tool3"}))

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		for _, tool := range result {
			if tool.Function.Name == "tool2" {
				t.Error("tool2 should have been filtered out")
			}
		}
	})
}

func TestChatTool_MarshalJSON(t *testing.T) {
	t.Run("should marshal chat tool correctly", func(t *testing.T) {
		tool := chatTool{
			Type: "function",
			Function: chatFunctionDef{
				Name:        "testFunction",
				Description: "A test function",
				Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
			},
		}

		data, err := json.Marshal(tool)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["type"] != "function" {
			t.Errorf("expected type 'function', got '%v'", result["type"])
		}

		fn := result["function"].(map[string]any)
		if fn["name"] != "testFunction" {
			t.Errorf("expected function name 'testFunction', got '%v'", fn["name"])
		}
		if fn["description"] != "A test function" {
			t.Errorf("expected description 'A test function', got '%v'", fn["description"])
		}
	})
}

// Note: ai-sdk prepareChatTools also tests:
// - toolChoice handling ('auto', 'required', 'none', {type: 'tool', toolName: '...'})
// - Strict mode pass-through (strict: true, strict: false, undefined)
// - Warning generation for unsupported provider tools
//
// goai's convertToChatTools does not handle these - toolChoice is set separately.
// Strict mode is not currently supported in goai's chat tools (unlike responses API).
//
// TODO: Consider adding strict mode support to match ai-sdk behavior:
// ai-sdk: {type: "function", function: {..., strict: true}}
// goai: currently no strict field in chatFunctionDef
