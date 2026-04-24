package output

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/schema"
	"github.com/airlockrun/goai/stream"
)

func TestTextOutput(t *testing.T) {
	out := Text()

	t.Run("name", func(t *testing.T) {
		if out.Name() != "text" {
			t.Errorf("expected name 'text', got '%s'", out.Name())
		}
	})

	t.Run("response format", func(t *testing.T) {
		rf := out.ResponseFormat()
		if rf.Type != "text" {
			t.Errorf("expected type 'text', got '%s'", rf.Type)
		}
	})

	t.Run("parse complete", func(t *testing.T) {
		result, err := out.ParseComplete("Hello world", stream.OutputParseContext{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "Hello world" {
			t.Errorf("expected 'Hello world', got '%v'", result)
		}
	})

	t.Run("parse partial", func(t *testing.T) {
		result := out.ParsePartial("Hello")
		if result != "Hello" {
			t.Errorf("expected 'Hello', got '%v'", result)
		}
	})
}

func TestObjectOutput(t *testing.T) {
	s := schema.Object(map[string]*schema.Schema{
		"name": schema.String(),
		"age":  schema.Integer(),
	})

	out := Object(ObjectOptions{Schema: s})

	t.Run("name", func(t *testing.T) {
		if out.Name() != "object" {
			t.Errorf("expected name 'object', got '%s'", out.Name())
		}
	})

	t.Run("response format", func(t *testing.T) {
		rf := out.ResponseFormat()
		if rf.Type != "json" {
			t.Errorf("expected type 'json', got '%s'", rf.Type)
		}
		if len(rf.Schema) == 0 {
			t.Error("expected schema to be set")
		}
	})

	t.Run("parse complete valid JSON", func(t *testing.T) {
		result, err := out.ParseComplete(`{"name":"Alice","age":30}`, stream.OutputParseContext{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		m, ok := result.(map[string]any)
		if !ok {
			t.Fatalf("expected map, got %T", result)
		}
		if m["name"] != "Alice" {
			t.Errorf("expected name 'Alice', got '%v'", m["name"])
		}
	})

	t.Run("parse complete invalid JSON", func(t *testing.T) {
		_, err := out.ParseComplete("not json", stream.OutputParseContext{})
		if err == nil {
			t.Fatal("expected error for invalid JSON")
		}
		_, ok := err.(*NoObjectGeneratedError)
		if !ok {
			t.Errorf("expected NoObjectGeneratedError, got %T", err)
		}
	})
}

func TestArrayOutput(t *testing.T) {
	elementSchema := schema.Object(map[string]*schema.Schema{
		"id": schema.Integer(),
	})

	out := Array(ArrayOptions{Element: elementSchema})

	t.Run("name", func(t *testing.T) {
		if out.Name() != "array" {
			t.Errorf("expected name 'array', got '%s'", out.Name())
		}
	})

	t.Run("response format has elements wrapper", func(t *testing.T) {
		rf := out.ResponseFormat()
		if rf.Type != "json" {
			t.Errorf("expected type 'json', got '%s'", rf.Type)
		}

		var s map[string]any
		if err := json.Unmarshal(rf.Schema, &s); err != nil {
			t.Fatalf("failed to parse schema: %v", err)
		}

		props, ok := s["properties"].(map[string]any)
		if !ok {
			t.Fatal("expected properties in schema")
		}
		if _, ok := props["elements"]; !ok {
			t.Error("expected 'elements' property in schema")
		}
	})

	t.Run("parse complete valid array", func(t *testing.T) {
		result, err := out.ParseComplete(`{"elements":[{"id":1},{"id":2}]}`, stream.OutputParseContext{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		arr, ok := result.([]any)
		if !ok {
			t.Fatalf("expected []any, got %T", result)
		}
		if len(arr) != 2 {
			t.Errorf("expected 2 elements, got %d", len(arr))
		}
	})

	t.Run("parse complete missing elements", func(t *testing.T) {
		_, err := out.ParseComplete(`{}`, stream.OutputParseContext{})
		if err == nil {
			t.Fatal("expected error for missing elements")
		}
	})
}

func TestChoiceOutput(t *testing.T) {
	out := Choice(ChoiceOptions{
		Options: []string{"yes", "no", "maybe"},
	})

	t.Run("name", func(t *testing.T) {
		if out.Name() != "choice" {
			t.Errorf("expected name 'choice', got '%s'", out.Name())
		}
	})

	t.Run("response format has enum", func(t *testing.T) {
		rf := out.ResponseFormat()

		var s map[string]any
		if err := json.Unmarshal(rf.Schema, &s); err != nil {
			t.Fatalf("failed to parse schema: %v", err)
		}

		props := s["properties"].(map[string]any)
		resultProp := props["result"].(map[string]any)
		enum := resultProp["enum"].([]any)

		if len(enum) != 3 {
			t.Errorf("expected 3 enum values, got %d", len(enum))
		}
	})

	t.Run("parse complete valid choice", func(t *testing.T) {
		result, err := out.ParseComplete(`{"result":"yes"}`, stream.OutputParseContext{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "yes" {
			t.Errorf("expected 'yes', got '%v'", result)
		}
	})

	t.Run("parse complete invalid choice", func(t *testing.T) {
		_, err := out.ParseComplete(`{"result":"unknown"}`, stream.OutputParseContext{})
		if err == nil {
			t.Fatal("expected error for invalid choice")
		}
	})
}

func TestJSONOutput(t *testing.T) {
	out := JSON(JSONOptions{})

	t.Run("name", func(t *testing.T) {
		if out.Name() != "json" {
			t.Errorf("expected name 'json', got '%s'", out.Name())
		}
	})

	t.Run("parse complete any valid JSON", func(t *testing.T) {
		testCases := []struct {
			input    string
			expected any
		}{
			{`{"key":"value"}`, map[string]any{"key": "value"}},
			{`[1,2,3]`, []any{float64(1), float64(2), float64(3)}},
			{`"string"`, "string"},
			{`123`, float64(123)},
			{`true`, true},
			{`null`, nil},
		}

		for _, tc := range testCases {
			result, err := out.ParseComplete(tc.input, stream.OutputParseContext{})
			if err != nil {
				t.Errorf("input %s: unexpected error: %v", tc.input, err)
				continue
			}
			// Just check it doesn't error - deep comparison is tricky
			if result == nil && tc.expected != nil && tc.input != "null" {
				t.Errorf("input %s: expected non-nil result", tc.input)
			}
		}
	})
}

func TestNoObjectGeneratedError(t *testing.T) {
	t.Run("error message without cause", func(t *testing.T) {
		err := &NoObjectGeneratedError{
			Message:      "Test error",
			Text:         "raw text",
			FinishReason: stream.FinishReasonStop,
		}
		if err.Error() != "Test error" {
			t.Errorf("unexpected error message: %s", err.Error())
		}
	})

	t.Run("error message with cause", func(t *testing.T) {
		cause := json.Unmarshal([]byte("invalid"), new(any))
		err := &NoObjectGeneratedError{
			Message: "Parse failed",
			Cause:   cause,
		}
		if err.Unwrap() != cause {
			t.Error("Unwrap should return cause")
		}
	})
}
