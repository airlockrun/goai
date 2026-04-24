package tool

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/errors"
)

// Tests for parseToolCall - translated from ai-sdk
// Source: ai-sdk/packages/ai/src/generate-text/parse-tool-call.test.ts

func TestParseToolCall(t *testing.T) {
	t.Run("should successfully parse a valid tool call", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      `{"param1": "test", "param2": 42}`,
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {"param1": {"type": "string"}, "param2": {"type": "number"}}}`),
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got invalid: %v", result.Error)
		}
		if result.ToolName != "testTool" {
			t.Errorf("expected tool name 'testTool', got '%s'", result.ToolName)
		}
		if result.ToolCallID != "123" {
			t.Errorf("expected tool call ID '123', got '%s'", result.ToolCallID)
		}

		input, ok := result.Input.(map[string]any)
		if !ok {
			t.Fatalf("expected map input, got %T", result.Input)
		}
		if input["param1"] != "test" {
			t.Errorf("expected param1 'test', got '%v'", input["param1"])
		}
		if input["param2"] != float64(42) {
			t.Errorf("expected param2 42, got '%v'", input["param2"])
		}
	})

	t.Run("should successfully parse a valid provider-executed dynamic tool call", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:             "tool-call",
				ToolName:         "testTool",
				ToolCallID:       "123",
				Input:            `{"param1": "test", "param2": 42}`,
				ProviderExecuted: true,
				Dynamic:          true,
				ProviderMetadata: map[string]any{
					"testProvider": map[string]any{"signature": "sig"},
				},
			},
			Tools: Set{}, // Empty tools - provider executed
		})

		if result.Invalid {
			t.Errorf("expected valid result, got invalid: %v", result.Error)
		}
		if !result.Dynamic {
			t.Error("expected dynamic to be true")
		}
		if !result.ProviderExecuted {
			t.Error("expected providerExecuted to be true")
		}
		if result.ProviderMetadata == nil {
			t.Error("expected providerMetadata to be set")
		}
	})

	t.Run("should successfully parse a valid tool call with provider metadata", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      `{"param1": "test", "param2": 42}`,
				ProviderMetadata: map[string]any{
					"testProvider": map[string]any{"signature": "sig"},
				},
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got invalid: %v", result.Error)
		}
		if result.ProviderMetadata == nil {
			t.Error("expected providerMetadata to be set")
		}
	})

	t.Run("should successfully process empty tool calls for tools that have no inputSchema", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "", // Empty input
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got invalid: %v", result.Error)
		}

		input, ok := result.Input.(map[string]any)
		if !ok {
			t.Fatalf("expected map input, got %T", result.Input)
		}
		if len(input) != 0 {
			t.Errorf("expected empty input, got %v", input)
		}
	})

	t.Run("should successfully process empty object tool calls for tools that have no inputSchema", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "{}",
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got invalid: %v", result.Error)
		}

		input, ok := result.Input.(map[string]any)
		if !ok {
			t.Fatalf("expected map input, got %T", result.Input)
		}
		if len(input) != 0 {
			t.Errorf("expected empty input, got %v", input)
		}
	})

	t.Run("should return NoSuchToolError when tools is nil", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "{}",
			},
			Tools: nil,
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}
		if !result.Dynamic {
			t.Error("expected dynamic to be true for invalid result")
		}
		if !errors.IsNoSuchToolError(result.Error) {
			t.Errorf("expected NoSuchToolError, got %T: %v", result.Error, result.Error)
		}
	})

	t.Run("should return NoSuchToolError when tool is not found", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "nonExistentTool",
				ToolCallID: "123",
				Input:      "{}",
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}
		if !errors.IsNoSuchToolError(result.Error) {
			t.Errorf("expected NoSuchToolError, got %T: %v", result.Error, result.Error)
		}

		// Check that available tools are included in error
		noSuchToolErr := result.Error.(*errors.NoSuchToolError)
		if len(noSuchToolErr.AvailableTools) != 1 || noSuchToolErr.AvailableTools[0] != "testTool" {
			t.Errorf("expected available tools [testTool], got %v", noSuchToolErr.AvailableTools)
		}
	})

	t.Run("should return InvalidToolInputError when input is invalid JSON", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "invalid json",
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}
		if !errors.IsInvalidToolInputError(result.Error) {
			t.Errorf("expected InvalidToolInputError, got %T: %v", result.Error, result.Error)
		}
	})
}

func TestParseToolCall_ToolCallRepair(t *testing.T) {
	t.Run("should invoke repairToolCall when provided and use its result", func(t *testing.T) {
		repairCalled := false

		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "invalid json", // This will trigger repair
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
			RepairToolCall: func(ctx RepairToolCallContext) (*RawToolCall, error) {
				repairCalled = true

				// Verify context
				if ctx.ToolCall.ToolName != "testTool" {
					t.Errorf("expected tool name 'testTool', got '%s'", ctx.ToolCall.ToolName)
				}
				if !errors.IsInvalidToolInputError(ctx.Error) {
					t.Errorf("expected InvalidToolInputError, got %T", ctx.Error)
				}

				// Return repaired tool call
				return &RawToolCall{
					Type:       "tool-call",
					ToolName:   "testTool",
					ToolCallID: "123",
					Input:      `{"param1": "test", "param2": 42}`,
				}, nil
			},
		})

		if !repairCalled {
			t.Error("expected repair function to be called")
		}
		if result.Invalid {
			t.Errorf("expected valid result after repair, got error: %v", result.Error)
		}

		input, ok := result.Input.(map[string]any)
		if !ok {
			t.Fatalf("expected map input, got %T", result.Input)
		}
		if input["param1"] != "test" {
			t.Errorf("expected param1 'test', got '%v'", input["param1"])
		}
	})

	t.Run("should re-throw error if tool call repair returns nil", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "invalid json",
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
			RepairToolCall: func(ctx RepairToolCallContext) (*RawToolCall, error) {
				return nil, nil // Repair unsuccessful
			},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}
		// Should have original error
		if !errors.IsInvalidToolInputError(result.Error) {
			t.Errorf("expected InvalidToolInputError, got %T: %v", result.Error, result.Error)
		}
	})

	t.Run("should return ToolCallRepairError if repairToolCall throws", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "testTool",
				ToolCallID: "123",
				Input:      "invalid json",
			},
			Tools: Set{
				"testTool": Tool{
					Name:        "testTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
			RepairToolCall: func(ctx RepairToolCallContext) (*RawToolCall, error) {
				return nil, &testError{"test error"}
			},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}
		if !errors.IsToolCallRepairError(result.Error) {
			t.Errorf("expected ToolCallRepairError, got %T: %v", result.Error, result.Error)
		}
	})
}

type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}

func TestParseToolCall_DynamicTools(t *testing.T) {
	t.Run("should handle dynamic tool in tools set", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "dynamicTool",
				ToolCallID: "123",
				Input:      `{"param1": "test"}`,
			},
			Tools: Set{
				"dynamicTool": Tool{
					Name:        "dynamicTool",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got error: %v", result.Error)
		}
		if result.ToolName != "dynamicTool" {
			t.Errorf("expected tool name 'dynamicTool', got '%s'", result.ToolName)
		}
	})

	t.Run("should handle provider-executed dynamic tool not in tools set", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:             "tool-call",
				ToolName:         "providerTool",
				ToolCallID:       "123",
				Input:            `{"query": "search term"}`,
				ProviderExecuted: true,
				Dynamic:          true,
			},
			Tools: Set{
				"localTool": Tool{
					Name: "localTool",
				},
			},
		})

		if result.Invalid {
			t.Errorf("expected valid result, got error: %v", result.Error)
		}
		if result.ToolName != "providerTool" {
			t.Errorf("expected tool name 'providerTool', got '%s'", result.ToolName)
		}
		if !result.ProviderExecuted {
			t.Error("expected providerExecuted to be true")
		}
		if !result.Dynamic {
			t.Error("expected dynamic to be true")
		}
	})
}

func TestParseToolCall_InputParsedForErrors(t *testing.T) {
	t.Run("should include parsed input in error result when possible", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "nonExistentTool",
				ToolCallID: "123",
				Input:      `{"key": "value"}`,
			},
			Tools: Set{},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}

		// Input should be parsed JSON even though there's an error
		input, ok := result.Input.(map[string]any)
		if !ok {
			t.Fatalf("expected map input, got %T", result.Input)
		}
		if input["key"] != "value" {
			t.Errorf("expected key 'value', got '%v'", input["key"])
		}
	})

	t.Run("should include raw input string when JSON parsing fails", func(t *testing.T) {
		result := ParseToolCall(ParseToolCallOptions{
			ToolCall: RawToolCall{
				Type:       "tool-call",
				ToolName:   "nonExistentTool",
				ToolCallID: "123",
				Input:      "not json",
			},
			Tools: Set{},
		})

		if !result.Invalid {
			t.Error("expected invalid result")
		}

		// Input should be the raw string
		if result.Input != "not json" {
			t.Errorf("expected raw input 'not json', got '%v'", result.Input)
		}
	})
}
