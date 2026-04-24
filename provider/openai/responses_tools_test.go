package openai

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for convertToResponsesTools - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/responses/openai-responses-prepare-tools.test.ts
//
// Note: ai-sdk has two tool types: 'function' and 'provider'
// - function: Regular tools with schema, description, strict mode
// - provider: OpenAI-specific tools (code_interpreter, web_search, etc.)
//
// goai currently only supports function tools.
// Provider tools (code_interpreter, image_generation, local_shell, web_search, apply_patch)
// are NOT implemented in goai yet.

func TestConvertToResponsesTools_FunctionToolsStrictMode(t *testing.T) {
	// Note: ai-sdk tests show strict mode should be passed through as-is:
	// - strict: true → include "strict": true
	// - strict: false → include "strict": false
	// - strict: undefined → don't include "strict" field
	//
	// Current goai implementation hardcodes Strict: true always, which differs from ai-sdk.
	// These tests document the expected ai-sdk behavior.

	t.Run("should pass through strict mode when strict is true", func(t *testing.T) {
		tools := tool.Set{
			"testFunction": {
				Name:        "testFunction",
				Description: "A test function",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
			},
		}

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		fn, ok := result[0].(responsesTool)
		if !ok {
			t.Fatalf("expected responsesTool, got %T", result[0])
		}
		if fn.Type != "function" {
			t.Errorf("expected type 'function', got '%s'", fn.Type)
		}
		if fn.Name != "testFunction" {
			t.Errorf("expected name 'testFunction', got '%s'", fn.Name)
		}
		if fn.Description != "A test function" {
			t.Errorf("expected description 'A test function', got '%s'", fn.Description)
		}
		// Strict is now a pointer to allow explicit false values
		if fn.Strict == nil || !*fn.Strict {
			t.Errorf("expected strict to be true, got %v", fn.Strict)
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

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		// Find each tool by name (order not guaranteed due to map iteration)
		var foundTool1, foundTool2 bool
		for _, item := range result {
			fn, ok := item.(responsesTool)
			if !ok {
				t.Fatalf("expected responsesTool, got %T", item)
			}
			if fn.Name == "tool1" {
				foundTool1 = true
				if fn.Description != "First tool" {
					t.Errorf("tool1: expected description 'First tool', got '%s'", fn.Description)
				}
			}
			if fn.Name == "tool2" {
				foundTool2 = true
				if fn.Description != "Second tool" {
					t.Errorf("tool2: expected description 'Second tool', got '%s'", fn.Description)
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

		// Only include tool1 and tool3
		result := convertToResponsesTools(tools.Ordered([]string{"tool1", "tool3"}), true)

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		for _, item := range result {
			fn, ok := item.(responsesTool)
			if !ok {
				t.Fatalf("expected responsesTool, got %T", item)
			}
			if fn.Name == "tool2" {
				t.Error("tool2 should have been filtered out")
			}
		}
	})

	t.Run("should return empty array when no tools", func(t *testing.T) {
		tools := tool.Set{}

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 0 {
			t.Fatalf("expected 0 tools, got %d", len(result))
		}
	})
}

func TestConvertToResponsesTools_SchemaProcessing(t *testing.T) {
	t.Run("should add additionalProperties false for strict mode", func(t *testing.T) {
		tools := tool.Set{
			"testFunction": {
				Name:        "testFunction",
				Description: "A test function",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {"name": {"type": "string"}}}`),
			},
		}

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		fn, ok := result[0].(responsesTool)
		if !ok {
			t.Fatalf("expected responsesTool, got %T", result[0])
		}

		// Parse the parameters to check additionalProperties
		var params map[string]any
		if err := json.Unmarshal(fn.Parameters, &params); err != nil {
			t.Fatalf("failed to parse parameters: %v", err)
		}

		additionalProps, ok := params["additionalProperties"]
		if !ok {
			t.Error("expected additionalProperties to be set")
		} else if additionalProps != false {
			t.Errorf("expected additionalProperties to be false, got %v", additionalProps)
		}
	})

	t.Run("should ensure all properties are in required array", func(t *testing.T) {
		tools := tool.Set{
			"testFunction": {
				Name:        "testFunction",
				Description: "A test function",
				InputSchema: json.RawMessage(`{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}`),
			},
		}

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		fn, ok := result[0].(responsesTool)
		if !ok {
			t.Fatalf("expected responsesTool, got %T", result[0])
		}

		var params map[string]any
		if err := json.Unmarshal(fn.Parameters, &params); err != nil {
			t.Fatalf("failed to parse parameters: %v", err)
		}

		required, ok := params["required"].([]any)
		if !ok {
			t.Fatal("expected required to be an array")
		}

		requiredSet := make(map[string]bool)
		for _, r := range required {
			if s, ok := r.(string); ok {
				requiredSet[s] = true
			}
		}

		if !requiredSet["name"] {
			t.Error("expected 'name' to be in required")
		}
		if !requiredSet["age"] {
			t.Error("expected 'age' to be in required")
		}
	})

	t.Run("should recursively process nested objects", func(t *testing.T) {
		tools := tool.Set{
			"testFunction": {
				Name:        "testFunction",
				Description: "A test function",
				InputSchema: json.RawMessage(`{
					"type": "object",
					"properties": {
						"user": {
							"type": "object",
							"properties": {
								"name": {"type": "string"}
							}
						}
					}
				}`),
			},
		}

		result := convertToResponsesTools(tools.Ordered(nil), true)

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		fn, ok := result[0].(responsesTool)
		if !ok {
			t.Fatalf("expected responsesTool, got %T", result[0])
		}

		var params map[string]any
		if err := json.Unmarshal(fn.Parameters, &params); err != nil {
			t.Fatalf("failed to parse parameters: %v", err)
		}

		props := params["properties"].(map[string]any)
		user := props["user"].(map[string]any)

		// Nested object should also have additionalProperties: false
		additionalProps, ok := user["additionalProperties"]
		if !ok {
			t.Error("expected nested object to have additionalProperties")
		} else if additionalProps != false {
			t.Errorf("expected nested additionalProperties to be false, got %v", additionalProps)
		}
	})
}

func TestResponsesTool_MarshalJSON(t *testing.T) {
	t.Run("should marshal function tool correctly", func(t *testing.T) {
		strictTrue := true
		tool := responsesTool{
			Type:        "function",
			Name:        "testFunction",
			Description: "A test function",
			Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
			Strict:      &strictTrue,
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
		if result["name"] != "testFunction" {
			t.Errorf("expected name 'testFunction', got '%v'", result["name"])
		}
		if result["strict"] != true {
			t.Errorf("expected strict true, got '%v'", result["strict"])
		}
	})
}

// TODO: The following tests cannot be translated yet because goai doesn't support
// provider tools. These should be added when provider tool support is implemented.
//
// Provider tools from ai-sdk:
// - openai.code_interpreter
// - openai.image_generation
// - openai.local_shell
// - openai.web_search
// - openai.apply_patch
// - openai.file_search
// - openai.mcp
//
// See: ai-sdk/packages/openai/src/responses/openai-responses-prepare-tools.test.ts
