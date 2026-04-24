package errors

import (
	"fmt"
)

// APICallError represents an error from an API call with full context.
// This is a Go translation of ai-sdk/packages/provider/src/errors/api-call-error.ts
type APICallError struct {
	// Message is the error message.
	Message string

	// URL is the request URL.
	URL string

	// RequestBodyValues contains the request body values for debugging.
	RequestBodyValues any

	// StatusCode is the HTTP status code (0 if not applicable).
	StatusCode int

	// ResponseHeaders are the response headers.
	ResponseHeaders map[string]string

	// ResponseBody is the raw response body.
	ResponseBody string

	// Data is the parsed error data from the provider.
	Data any

	// Cause is the underlying error.
	Cause error

	// IsRetryable indicates if the error is retryable.
	IsRetryable bool
}

// Error implements the error interface.
func (e *APICallError) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("API call error (status %d): %s", e.StatusCode, e.Message)
	}
	return fmt.Sprintf("API call error: %s", e.Message)
}

// Unwrap returns the underlying cause for errors.Is/As compatibility.
func (e *APICallError) Unwrap() error {
	return e.Cause
}

// NewAPICallError creates a new APICallError with default retryability logic.
// By default, errors are retryable if the status code is:
// - 408 (Request Timeout)
// - 409 (Conflict)
// - 429 (Too Many Requests)
// - >= 500 (Server Error)
func NewAPICallError(opts APICallErrorOptions) *APICallError {
	isRetryable := opts.IsRetryable
	if !opts.IsRetryableSet && opts.StatusCode != 0 {
		isRetryable = opts.StatusCode == 408 ||
			opts.StatusCode == 409 ||
			opts.StatusCode == 429 ||
			opts.StatusCode >= 500
	}

	return &APICallError{
		Message:           opts.Message,
		URL:               opts.URL,
		RequestBodyValues: opts.RequestBodyValues,
		StatusCode:        opts.StatusCode,
		ResponseHeaders:   opts.ResponseHeaders,
		ResponseBody:      opts.ResponseBody,
		Data:              opts.Data,
		Cause:             opts.Cause,
		IsRetryable:       isRetryable,
	}
}

// APICallErrorOptions contains options for creating an APICallError.
type APICallErrorOptions struct {
	Message           string
	URL               string
	RequestBodyValues any
	StatusCode        int
	ResponseHeaders   map[string]string
	ResponseBody      string
	Data              any
	Cause             error
	IsRetryable       bool
	IsRetryableSet    bool // true if IsRetryable was explicitly set
}
