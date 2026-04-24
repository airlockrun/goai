package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for tool conversion - translated from ai-sdk
// Source: ai-sdk/packages/anthropic/src/anthropic-prepare-tools.test.ts
//
// Note: ai-sdk has provider tools (e.g., anthropic.web_search, anthropic.computer_use)
// which are NOT implemented in goai yet.

func TestConvertToAnthropicTools(t *testing.T) {
	t.Run("should return empty array when no tools", func(t *testing.T) {
		tools := tool.Set{}

		result, _ := convertToAnthropicTools(tools.Ordered(nil))

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

		result, _ := convertToAnthropicTools(tools.Ordered(nil))

		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		t0 := result[0].(anthropicTool)
		if t0.Name != "testFunction" {
			t.Errorf("expected name 'testFunction', got '%s'", t0.Name)
		}
		if t0.Description != "A test function" {
			t.Errorf("expected description 'A test function', got '%s'", t0.Description)
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

		result, _ := convertToAnthropicTools(tools.Ordered(nil))

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		// Find each tool
		var foundTool1, foundTool2 bool
		for _, tt := range result {
			ft, ok := tt.(anthropicTool)
			if !ok {
				continue
			}
			if ft.Name == "tool1" {
				foundTool1 = true
			}
			if ft.Name == "tool2" {
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

	t.Run("should pass through InputExamples as input_examples", func(t *testing.T) {
		tools := tool.Set{
			"weather": {
				Name:        "weather",
				Description: "Get the weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"location":"Berlin"}`)},
					{Input: json.RawMessage(`{"location":"Paris"}`)},
				},
			},
		}
		result, _ := convertToAnthropicTools(tools.Ordered(nil))
		if len(result) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(result))
		}
		t0 := result[0].(anthropicTool)
		examples := t0.InputExamples
		if len(examples) != 2 {
			t.Fatalf("InputExamples length = %d, want 2", len(examples))
		}
		if string(examples[0].Input) != `{"location":"Berlin"}` {
			t.Errorf("examples[0].Input = %s, want Berlin", string(examples[0].Input))
		}

		// Round-trip through JSON to confirm wire key is input_examples.
		raw, err := json.Marshal(t0)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, ok := decoded["input_examples"]; !ok {
			t.Errorf("marshaled tool missing input_examples key: %s", string(raw))
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

		result, _ := convertToAnthropicTools(tools.Ordered([]string{"tool1", "tool3"}))

		if len(result) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(result))
		}

		for _, tt := range result {
			if ft, ok := tt.(anthropicTool); ok && ft.Name == "tool2" {
				t.Error("tool2 should have been filtered out")
			}
		}
	})
}

func TestAnthropicTool_MarshalJSON(t *testing.T) {
	t.Run("should marshal tool correctly", func(t *testing.T) {
		tool := anthropicTool{
			Name:        "testFunction",
			Description: "A test function",
			InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
		}

		data, err := json.Marshal(tool)
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
// - anthropic.web_search (with domain filters)
// - anthropic.web_fetch
// - anthropic.computer_use
// - anthropic.text_editor
// - anthropic.bash
//
// These are NOT implemented in goai yet.
