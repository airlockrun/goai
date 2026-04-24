package errors

import (
	"errors"
	"fmt"
	"testing"
)

// Tests for tool call error types - translated from ai-sdk
// Source: ai-sdk/packages/ai/src/error/no-such-tool-error.ts
// Source: ai-sdk/packages/ai/src/error/invalid-tool-input-error.ts
// Source: ai-sdk/packages/ai/src/error/tool-call-repair-error.ts

func TestNoSuchToolError(t *testing.T) {
	t.Run("should have correct message when no tools available", func(t *testing.T) {
		err := &NoSuchToolError{
			ToolName: "testTool",
		}

		expected := "Model tried to call unavailable tool 'testTool'. No tools are available."
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should have correct message with available tools", func(t *testing.T) {
		err := &NoSuchToolError{
			ToolName:       "nonExistentTool",
			AvailableTools: []string{"tool1", "tool2"},
		}

		expected := "Model tried to call unavailable tool 'nonExistentTool'. Available tools: tool1, tool2."
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should be checkable with errors.Is", func(t *testing.T) {
		err := &NoSuchToolError{
			ToolName: "testTool",
		}

		if !errors.Is(err, ErrNoSuchTool) {
			t.Error("expected errors.Is(err, ErrNoSuchTool) to be true")
		}
	})

	t.Run("should be checkable with IsNoSuchToolError", func(t *testing.T) {
		err := &NoSuchToolError{
			ToolName: "testTool",
		}

		if !IsNoSuchToolError(err) {
			t.Error("expected IsNoSuchToolError to return true")
		}

		// Wrapped error should also work
		wrapped := fmt.Errorf("wrapped: %w", err)
		if !IsNoSuchToolError(wrapped) {
			t.Error("expected IsNoSuchToolError to return true for wrapped error")
		}

		// Non-matching error
		other := errors.New("some other error")
		if IsNoSuchToolError(other) {
			t.Error("expected IsNoSuchToolError to return false for non-matching error")
		}
	})
}

func TestInvalidToolInputError(t *testing.T) {
	t.Run("should have correct message", func(t *testing.T) {
		err := &InvalidToolInputError{
			ToolName:  "testTool",
			ToolInput: `{"invalid": "input"}`,
			Cause:     errors.New("missing required field 'name'"),
		}

		expected := "Invalid input for tool testTool: missing required field 'name'"
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should handle nil cause", func(t *testing.T) {
		err := &InvalidToolInputError{
			ToolName:  "testTool",
			ToolInput: `{"invalid": "input"}`,
			Cause:     nil,
		}

		expected := "Invalid input for tool testTool: "
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should be checkable with errors.Is", func(t *testing.T) {
		err := &InvalidToolInputError{
			ToolName: "testTool",
			Cause:    errors.New("validation failed"),
		}

		if !errors.Is(err, ErrInvalidToolInput) {
			t.Error("expected errors.Is(err, ErrInvalidToolInput) to be true")
		}
	})

	t.Run("should be checkable with IsInvalidToolInputError", func(t *testing.T) {
		err := &InvalidToolInputError{
			ToolName: "testTool",
		}

		if !IsInvalidToolInputError(err) {
			t.Error("expected IsInvalidToolInputError to return true")
		}

		// Wrapped error should also work
		wrapped := fmt.Errorf("wrapped: %w", err)
		if !IsInvalidToolInputError(wrapped) {
			t.Error("expected IsInvalidToolInputError to return true for wrapped error")
		}
	})
}

func TestToolCallRepairError(t *testing.T) {
	t.Run("should have correct message", func(t *testing.T) {
		err := &ToolCallRepairError{
			Cause:         errors.New("repair failed"),
			OriginalError: errors.New("original error"),
		}

		expected := "Error repairing tool call: repair failed"
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should handle nil cause", func(t *testing.T) {
		err := &ToolCallRepairError{
			OriginalError: errors.New("original error"),
		}

		expected := "Error repairing tool call"
		if err.Error() != expected {
			t.Errorf("got %q, want %q", err.Error(), expected)
		}
	})

	t.Run("should be checkable with errors.Is", func(t *testing.T) {
		err := &ToolCallRepairError{
			Cause: errors.New("repair failed"),
		}

		if !errors.Is(err, ErrToolCallRepair) {
			t.Error("expected errors.Is(err, ErrToolCallRepair) to be true")
		}
	})

	t.Run("should be checkable with IsToolCallRepairError", func(t *testing.T) {
		err := &ToolCallRepairError{
			Cause: errors.New("repair failed"),
		}

		if !IsToolCallRepairError(err) {
			t.Error("expected IsToolCallRepairError to return true")
		}

		// Wrapped error should also work
		wrapped := fmt.Errorf("wrapped: %w", err)
		if !IsToolCallRepairError(wrapped) {
			t.Error("expected IsToolCallRepairError to return true for wrapped error")
		}
	})
}

func TestJoinStrings(t *testing.T) {
	testCases := []struct {
		name     string
		strs     []string
		sep      string
		expected string
	}{
		{"empty slice", []string{}, ", ", ""},
		{"single element", []string{"a"}, ", ", "a"},
		{"two elements", []string{"a", "b"}, ", ", "a, b"},
		{"three elements", []string{"tool1", "tool2", "tool3"}, ", ", "tool1, tool2, tool3"},
		{"different separator", []string{"a", "b", "c"}, "-", "a-b-c"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := joinStrings(tc.strs, tc.sep)
			if result != tc.expected {
				t.Errorf("got %q, want %q", result, tc.expected)
			}
		})
	}
}
