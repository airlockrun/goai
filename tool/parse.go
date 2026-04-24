package tool

import (
	"encoding/json"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
)

// RawToolCall represents a tool call from the language model before validation.
// Source: ai-sdk/packages/provider/src/language-model-v3.ts (LanguageModelV3ToolCall)
type RawToolCall struct {
	Type       string `json:"type"` // "tool-call"
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	Input      string `json:"input"` // JSON string of the input
	// Dynamic indicates the tool is not statically defined
	Dynamic bool `json:"dynamic,omitempty"`
	// ProviderExecuted indicates the tool was executed by the provider
	ProviderExecuted bool `json:"providerExecuted,omitempty"`
	// ProviderMetadata contains provider-specific metadata
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

// ParsedToolCall represents a validated and parsed tool call.
// Source: ai-sdk/packages/ai/src/generate-text/tool-call.ts
type ParsedToolCall struct {
	Type       string `json:"type"` // "tool-call"
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	// Input is the parsed JSON input (may be any JSON value)
	Input any `json:"input"`
	// Title is the optional tool title
	Title string `json:"title,omitempty"`
	// Dynamic indicates this is a dynamic tool call
	Dynamic bool `json:"dynamic,omitempty"`
	// Invalid indicates the tool call failed validation
	Invalid bool `json:"invalid,omitempty"`
	// Error contains the validation error if Invalid is true
	Error error `json:"error,omitempty"`
	// ProviderExecuted indicates the tool was executed by the provider
	ProviderExecuted bool `json:"providerExecuted,omitempty"`
	// ProviderMetadata contains provider-specific metadata
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

// ParseToolCallOptions contains options for ParseToolCall.
type ParseToolCallOptions struct {
	// ToolCall is the raw tool call from the model.
	ToolCall RawToolCall
	// Tools is the set of available tools. May be nil.
	Tools Set
	// RepairToolCall is an optional function to repair invalid tool calls.
	RepairToolCall RepairToolCallFunc
	// System is the system message (for repair context).
	System string
	// Messages is the conversation history (for repair context).
	Messages []message.Message
}

// RepairToolCallContext contains context for the repair function.
type RepairToolCallContext struct {
	// ToolCall is the invalid tool call.
	ToolCall RawToolCall
	// Tools is the set of available tools.
	Tools Set
	// InputSchema returns the JSON schema for a tool by name.
	InputSchema func(toolName string) json.RawMessage
	// System is the system message.
	System string
	// Messages is the conversation history.
	Messages []message.Message
	// Error is the validation error that triggered repair.
	Error error
}

// RepairToolCallFunc is a function that attempts to repair an invalid tool call.
// Return nil to indicate the repair was unsuccessful.
type RepairToolCallFunc func(ctx RepairToolCallContext) (*RawToolCall, error)

// ParseToolCall validates and parses a tool call from the model.
// This mirrors ai-sdk's parseToolCall function.
//
// Source: ai-sdk/packages/ai/src/generate-text/parse-tool-call.ts
func ParseToolCall(opts ParseToolCallOptions) *ParsedToolCall {
	toolCall := opts.ToolCall
	tools := opts.Tools

	// Helper to create an invalid result
	makeInvalidResult := func(input any, err error, title string) *ParsedToolCall {
		return &ParsedToolCall{
			Type:             "tool-call",
			ToolCallID:       toolCall.ToolCallID,
			ToolName:         toolCall.ToolName,
			Input:            input,
			Dynamic:          true,
			Invalid:          true,
			Error:            err,
			Title:            title,
			ProviderExecuted: toolCall.ProviderExecuted,
			ProviderMetadata: toolCall.ProviderMetadata,
		}
	}

	// Get title for error results
	getTitle := func() string {
		if tools != nil {
			if t, ok := tools[toolCall.ToolName]; ok {
				// Tool doesn't have a title field currently, but we prepare for it
				_ = t
			}
		}
		return ""
	}

	// Parse input for error results
	parseInputForError := func() any {
		var parsed any
		if err := json.Unmarshal([]byte(toolCall.Input), &parsed); err == nil {
			return parsed
		}
		return toolCall.Input
	}

	// Handle case where no tools are provided
	if len(tools) == 0 {
		// Provider-executed dynamic tools don't need to be in our tool set
		if toolCall.ProviderExecuted && toolCall.Dynamic {
			return parseProviderExecutedDynamicToolCall(toolCall)
		}

		err := &errors.NoSuchToolError{
			ToolName: toolCall.ToolName,
		}
		return makeInvalidResult(parseInputForError(), err, getTitle())
	}

	// Try to parse the tool call
	result, parseErr := doParseToolCall(toolCall, tools)
	if parseErr == nil {
		return result
	}

	// If repair function is provided and error is repairable, try repair
	if opts.RepairToolCall != nil &&
		(errors.IsNoSuchToolError(parseErr) || errors.IsInvalidToolInputError(parseErr)) {

		repairCtx := RepairToolCallContext{
			ToolCall: toolCall,
			Tools:    tools,
			InputSchema: func(toolName string) json.RawMessage {
				if t, ok := tools[toolName]; ok {
					return t.InputSchema
				}
				return nil
			},
			System:   opts.System,
			Messages: opts.Messages,
			Error:    parseErr,
		}

		repairedToolCall, repairErr := opts.RepairToolCall(repairCtx)
		if repairErr != nil {
			// Repair function threw an error
			err := &errors.ToolCallRepairError{
				Cause:         repairErr,
				OriginalError: parseErr,
			}
			return makeInvalidResult(parseInputForError(), err, getTitle())
		}

		if repairedToolCall == nil {
			// Repair returned nil - use original error
			return makeInvalidResult(parseInputForError(), parseErr, getTitle())
		}

		// Try parsing the repaired tool call
		result, parseErr = doParseToolCall(*repairedToolCall, tools)
		if parseErr == nil {
			return result
		}
	}

	// Return invalid result with the error
	return makeInvalidResult(parseInputForError(), parseErr, getTitle())
}

// parseProviderExecutedDynamicToolCall handles tool calls that were executed by the provider.
func parseProviderExecutedDynamicToolCall(toolCall RawToolCall) *ParsedToolCall {
	var input any

	// Parse empty input as empty object
	if toolCall.Input == "" || toolCall.Input == "{}" {
		input = map[string]any{}
	} else {
		if err := json.Unmarshal([]byte(toolCall.Input), &input); err != nil {
			// Return invalid result for parse errors
			return &ParsedToolCall{
				Type:             "tool-call",
				ToolCallID:       toolCall.ToolCallID,
				ToolName:         toolCall.ToolName,
				Input:            toolCall.Input,
				Dynamic:          true,
				Invalid:          true,
				ProviderExecuted: true,
				ProviderMetadata: toolCall.ProviderMetadata,
				Error: &errors.InvalidToolInputError{
					ToolName:  toolCall.ToolName,
					ToolInput: toolCall.Input,
					Cause:     err,
				},
			}
		}
	}

	return &ParsedToolCall{
		Type:             "tool-call",
		ToolCallID:       toolCall.ToolCallID,
		ToolName:         toolCall.ToolName,
		Input:            input,
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: toolCall.ProviderMetadata,
	}
}

// doParseToolCall performs the actual parsing and validation.
func doParseToolCall(toolCall RawToolCall, tools Set) (*ParsedToolCall, error) {
	toolName := toolCall.ToolName
	_, exists := tools[toolName]

	if !exists {
		// Provider-executed dynamic tools don't need to be in our tool set
		if toolCall.ProviderExecuted && toolCall.Dynamic {
			result := parseProviderExecutedDynamicToolCall(toolCall)
			if result.Invalid {
				return nil, result.Error
			}
			return result, nil
		}

		return nil, &errors.NoSuchToolError{
			ToolName:       toolName,
			AvailableTools: tools.Names(),
		}
	}

	// Parse the input
	var input any

	// When input is empty, try passing empty object
	inputStr := toolCall.Input
	if inputStr == "" {
		inputStr = "{}"
	}

	if err := json.Unmarshal([]byte(inputStr), &input); err != nil {
		return nil, &errors.InvalidToolInputError{
			ToolName:  toolName,
			ToolInput: toolCall.Input,
			Cause:     err,
		}
	}

	// TODO: Add schema validation when we have a JSON schema validator
	// For now, we just parse the JSON and trust the model output.
	// In ai-sdk, this uses safeValidateTypes with the tool's inputSchema.

	return &ParsedToolCall{
		Type:             "tool-call",
		ToolCallID:       toolCall.ToolCallID,
		ToolName:         toolName,
		Input:            input,
		ProviderExecuted: toolCall.ProviderExecuted,
		ProviderMetadata: toolCall.ProviderMetadata,
		// Title would come from tool.Title if we had it
	}, nil
}

// ValidateToolInput validates that the input matches the tool's schema.
// This is a separate function for when you need explicit validation.
func ValidateToolInput(tool Tool, input any) error {
	// TODO: Implement proper JSON schema validation
	// For now, we just check that the input is valid JSON
	_, err := json.Marshal(input)
	return err
}
