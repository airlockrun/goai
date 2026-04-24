// Package errors provides custom error types for the goai package.
package errors

import (
	"errors"
	"fmt"
)

// Re-export standard library error functions for convenience.
var (
	Is     = errors.Is
	As     = errors.As
	New    = errors.New
	Unwrap = errors.Unwrap
	Join   = errors.Join
)

// Standard error types that can be checked with errors.Is.
var (
	// ErrAPIError indicates an error from the provider's API.
	ErrAPIError = errors.New("api error")

	// ErrRateLimited indicates the request was rate limited.
	ErrRateLimited = errors.New("rate limited")

	// ErrAuthenticationFailed indicates authentication failed.
	ErrAuthenticationFailed = errors.New("authentication failed")

	// ErrInvalidRequest indicates the request was invalid.
	ErrInvalidRequest = errors.New("invalid request")

	// ErrModelNotFound indicates the requested model was not found.
	ErrModelNotFound = errors.New("model not found")

	// ErrContentFiltered indicates content was filtered.
	ErrContentFiltered = errors.New("content filtered")

	// ErrContextLengthExceeded indicates the context length was exceeded.
	ErrContextLengthExceeded = errors.New("context length exceeded")

	// ErrTimeout indicates the request timed out.
	ErrTimeout = errors.New("request timeout")

	// ErrCanceled indicates the request was canceled.
	ErrCanceled = errors.New("request canceled")

	// ErrUnsupported indicates an unsupported operation.
	ErrUnsupported = errors.New("unsupported operation")

	// ErrInvalidResponse indicates an invalid response from the provider.
	ErrInvalidResponse = errors.New("invalid response")

	// ErrToolExecutionFailed indicates a tool execution failure.
	ErrToolExecutionFailed = errors.New("tool execution failed")

	// ErrJSONParseFailed indicates JSON parsing failed.
	ErrJSONParseFailed = errors.New("json parse failed")

	// ErrSchemaMismatch indicates the response didn't match the expected schema.
	ErrSchemaMismatch = errors.New("schema mismatch")
)

// APIError represents an error from a provider's API.
type APIError struct {
	// Provider is the provider that returned the error.
	Provider string

	// StatusCode is the HTTP status code (0 if not applicable).
	StatusCode int

	// Code is the error code from the provider.
	Code string

	// Message is the error message.
	Message string

	// Param is the parameter that caused the error (if applicable).
	Param string

	// Type is the error type from the provider.
	Type string

	// RequestID is the request ID for debugging.
	RequestID string

	// Retryable indicates if the error is retryable.
	Retryable bool

	// RetryAfter is the number of seconds to wait before retrying (0 if not specified).
	RetryAfter int
}

// Error implements the error interface.
func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s API error [%s]: %s", e.Provider, e.Code, e.Message)
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("%s API error (status %d): %s", e.Provider, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("%s API error: %s", e.Provider, e.Message)
}

// Unwrap returns the base error for errors.Is compatibility.
func (e *APIError) Unwrap() error {
	switch {
	case e.StatusCode == 401 || e.StatusCode == 403:
		return ErrAuthenticationFailed
	case e.StatusCode == 404:
		return ErrModelNotFound
	case e.StatusCode == 429:
		return ErrRateLimited
	case e.StatusCode == 400:
		return ErrInvalidRequest
	case e.Code == "content_filter":
		return ErrContentFiltered
	case e.Code == "context_length_exceeded":
		return ErrContextLengthExceeded
	default:
		return ErrAPIError
	}
}

// ToolError represents a tool execution error.
type ToolError struct {
	// ToolCallID is the ID of the tool call.
	ToolCallID string

	// ToolName is the name of the tool.
	ToolName string

	// Message is the error message.
	Message string

	// Cause is the underlying error.
	Cause error
}

// Error implements the error interface.
func (e *ToolError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("tool %q execution failed: %s: %v", e.ToolName, e.Message, e.Cause)
	}
	return fmt.Sprintf("tool %q execution failed: %s", e.ToolName, e.Message)
}

// Unwrap returns the underlying error.
func (e *ToolError) Unwrap() error {
	if e.Cause != nil {
		return e.Cause
	}
	return ErrToolExecutionFailed
}

// JSONParseError represents a JSON parsing error.
type JSONParseError struct {
	// Text is the text that failed to parse.
	Text string

	// Cause is the underlying error.
	Cause error
}

// Error implements the error interface.
func (e *JSONParseError) Error() string {
	if len(e.Text) > 100 {
		return fmt.Sprintf("failed to parse JSON: %v (text: %s...)", e.Cause, e.Text[:100])
	}
	return fmt.Sprintf("failed to parse JSON: %v (text: %s)", e.Cause, e.Text)
}

// Unwrap returns the underlying error.
func (e *JSONParseError) Unwrap() error {
	return ErrJSONParseFailed
}

// SchemaValidationError represents a schema validation error.
type SchemaValidationError struct {
	// Schema is a description of the expected schema.
	Schema string

	// Value is the value that didn't match.
	Value any

	// Path is the JSON path to the invalid field.
	Path string

	// Message is the validation error message.
	Message string
}

// Error implements the error interface.
func (e *SchemaValidationError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("schema validation failed at %s: %s", e.Path, e.Message)
	}
	return fmt.Sprintf("schema validation failed: %s", e.Message)
}

// Unwrap returns the base error.
func (e *SchemaValidationError) Unwrap() error {
	return ErrSchemaMismatch
}

// RetryInfo contains information about retry behavior.
type RetryInfo struct {
	// ShouldRetry indicates if the error is retryable.
	ShouldRetry bool

	// RetryAfter is the suggested wait time in seconds.
	RetryAfter int

	// Attempt is the current attempt number.
	Attempt int

	// MaxAttempts is the maximum number of attempts.
	MaxAttempts int
}

// GetRetryInfo extracts retry information from an error.
func GetRetryInfo(err error) *RetryInfo {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return &RetryInfo{
			ShouldRetry: apiErr.Retryable,
			RetryAfter:  apiErr.RetryAfter,
		}
	}
	return &RetryInfo{
		ShouldRetry: false,
	}
}

// IsRetryable returns true if the error is retryable.
func IsRetryable(err error) bool {
	info := GetRetryInfo(err)
	return info.ShouldRetry
}

// Tool call error types - translated from ai-sdk
// Source: ai-sdk/packages/ai/src/error/no-such-tool-error.ts
// Source: ai-sdk/packages/ai/src/error/invalid-tool-input-error.ts
// Source: ai-sdk/packages/ai/src/error/tool-call-repair-error.ts

// ErrNoSuchTool indicates the model tried to call a tool that doesn't exist.
var ErrNoSuchTool = errors.New("no such tool")

// ErrInvalidToolInput indicates the tool input doesn't match the expected schema.
var ErrInvalidToolInput = errors.New("invalid tool input")

// ErrToolCallRepair indicates the tool call repair function failed.
var ErrToolCallRepair = errors.New("tool call repair failed")

// NoSuchToolError is returned when the model tries to call a tool that doesn't exist.
type NoSuchToolError struct {
	// ToolName is the name of the tool the model tried to call.
	ToolName string

	// AvailableTools is the list of available tool names (nil if no tools available).
	AvailableTools []string
}

// Error implements the error interface.
func (e *NoSuchToolError) Error() string {
	if len(e.AvailableTools) == 0 {
		return fmt.Sprintf("Model tried to call unavailable tool '%s'. No tools are available.", e.ToolName)
	}
	return fmt.Sprintf("Model tried to call unavailable tool '%s'. Available tools: %s.",
		e.ToolName, joinStrings(e.AvailableTools, ", "))
}

// Unwrap returns the base error for errors.Is compatibility.
func (e *NoSuchToolError) Unwrap() error {
	return ErrNoSuchTool
}

// IsNoSuchToolError returns true if err is a NoSuchToolError.
func IsNoSuchToolError(err error) bool {
	var target *NoSuchToolError
	return errors.As(err, &target)
}

// InvalidToolInputError is returned when tool input doesn't match the expected schema.
type InvalidToolInputError struct {
	// ToolName is the name of the tool.
	ToolName string

	// ToolInput is the input that was invalid.
	ToolInput string

	// Cause is the underlying validation error.
	Cause error
}

// Error implements the error interface.
func (e *InvalidToolInputError) Error() string {
	causeMsg := ""
	if e.Cause != nil {
		causeMsg = e.Cause.Error()
	}
	return fmt.Sprintf("Invalid input for tool %s: %s", e.ToolName, causeMsg)
}

// Unwrap returns the base error for errors.Is compatibility.
func (e *InvalidToolInputError) Unwrap() error {
	return ErrInvalidToolInput
}

// IsInvalidToolInputError returns true if err is an InvalidToolInputError.
func IsInvalidToolInputError(err error) bool {
	var target *InvalidToolInputError
	return errors.As(err, &target)
}

// ToolCallRepairError is returned when the tool call repair function fails.
type ToolCallRepairError struct {
	// Cause is the error from the repair function.
	Cause error

	// OriginalError is the original error that triggered repair.
	OriginalError error
}

// Error implements the error interface.
func (e *ToolCallRepairError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("Error repairing tool call: %s", e.Cause.Error())
	}
	return "Error repairing tool call"
}

// Unwrap returns the base error for errors.Is compatibility.
func (e *ToolCallRepairError) Unwrap() error {
	return ErrToolCallRepair
}

// IsToolCallRepairError returns true if err is a ToolCallRepairError.
func IsToolCallRepairError(err error) bool {
	var target *ToolCallRepairError
	return errors.As(err, &target)
}

// ErrMissingToolResults indicates tool results are missing for tool calls.
var ErrMissingToolResults = errors.New("missing tool results")

// MissingToolResultsError is returned when tool results are missing for tool calls.
// Source: ai-sdk/packages/ai/src/error/missing-tool-result-error.ts
type MissingToolResultsError struct {
	// ToolCallIDs are the IDs of the tool calls that are missing results.
	ToolCallIDs []string
}

// Error implements the error interface.
func (e *MissingToolResultsError) Error() string {
	if len(e.ToolCallIDs) == 1 {
		return fmt.Sprintf("Tool result is missing for tool call %s.", e.ToolCallIDs[0])
	}
	return fmt.Sprintf("Tool results are missing for tool calls %s.", joinStrings(e.ToolCallIDs, ", "))
}

// Unwrap returns the base error for errors.Is compatibility.
func (e *MissingToolResultsError) Unwrap() error {
	return ErrMissingToolResults
}

// IsMissingToolResultsError returns true if err is a MissingToolResultsError.
func IsMissingToolResultsError(err error) bool {
	var target *MissingToolResultsError
	return errors.As(err, &target)
}

// joinStrings joins strings with a separator.
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	if len(strs) == 1 {
		return strs[0]
	}
	// Calculate total length
	n := len(sep) * (len(strs) - 1)
	for _, s := range strs {
		n += len(s)
	}
	// Build result
	b := make([]byte, n)
	pos := copy(b, strs[0])
	for _, s := range strs[1:] {
		pos += copy(b[pos:], sep)
		pos += copy(b[pos:], s)
	}
	return string(b)
}
