package google

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for tool conversion - translated from ai-sdk
// Source: ai-sdk/packages/google/src/google-prepare-tools.test.ts
//
// Note: ai-sdk has provider tools (e.g., google.google_search, google.url_context)
// which are NOT implemented in goai yet.

func TestConvertToGeminiFunctions(t *testing.T) {
	t.Run("should return empty array when no tools", func(t *testing.T) {
		tools := tool.Set{}

		result := convertToGeminiFunctions(tools.Ordered(nil))

		if len(result) != 0 {
			t.Fatalf("expected 0 functions, got %d", len(result))
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

		result := convertToGeminiFunctions(tools.Ordered(nil))

		if len(result) != 1 {
			t.Fatalf("expected 1 function, got %d", len(result))
		}
		if result[0].Name != "testFunction" {
			t.Errorf("expected name 'testFunction', got '%s'", result[0].Name)
		}
		if result[0].Description != "A test function" {
			t.Errorf("expected description 'A test function', got '%s'", result[0].Description)
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

		result := convertToGeminiFunctions(tools.Ordered(nil))

		if len(result) != 2 {
			t.Fatalf("expected 2 functions, got %d", len(result))
		}

		// Find each function
		var foundTool1, foundTool2 bool
		for _, fn := range result {
			if fn.Name == "tool1" {
				foundTool1 = true
			}
			if fn.Name == "tool2" {
				foundTool2 = true
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

		result := convertToGeminiFunctions(tools.Ordered([]string{"tool1", "tool3"}))

		if len(result) != 2 {
			t.Fatalf("expected 2 functions, got %d", len(result))
		}

		for _, fn := range result {
			if fn.Name == "tool2" {
				t.Error("tool2 should have been filtered out")
			}
		}
	})
}

func TestGeminiFunctionDeclaration_MarshalJSON(t *testing.T) {
	t.Run("should marshal function declaration correctly", func(t *testing.T) {
		fn := geminiFunctionDeclaration{
			Name:        "testFunction",
			Description: "A test function",
			Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
		}

		data, err := json.Marshal(fn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result["name"] != "testFunction" {
			t.Errorf("expected name 'testFunction', got '%v'", result["name"])
		}
		if result["description"] != "A test function" {
			t.Errorf("expected description 'A test function', got '%v'", result["description"])
		}
	})
}

// Note: ai-sdk also tests provider tools like:
// - google.google_search
// - google.url_context
// - google.file_search
// - google.code_execution
//
// These are NOT implemented in goai yet.
