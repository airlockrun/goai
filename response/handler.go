// Package response provides HTTP response handlers for API calls.
// This is a Go translation of ai-sdk/packages/provider-utils/src/response-handler.ts
package response

import (
	"net/http"
)

// HandlerOptions contains inputs for a response handler.
type HandlerOptions struct {
	// URL is the request URL.
	URL string

	// RequestBodyValues contains the request body values for error reporting.
	RequestBodyValues any

	// Response is the HTTP response to process.
	Response *http.Response
}

// HandlerResult contains the result of processing a response.
type HandlerResult[T any] struct {
	// Value is the parsed/processed response value.
	Value T

	// RawValue is the raw parsed value before schema validation (for JSON responses).
	RawValue any

	// ResponseHeaders are the response headers as a flat map.
	ResponseHeaders map[string]string
}

// Handler processes an HTTP response and returns a typed result.
type Handler[T any] func(opts HandlerOptions) (*HandlerResult[T], error)

// ExtractResponseHeaders converts http.Header to map[string]string.
// For headers with multiple values, only the first value is included.
func ExtractResponseHeaders(resp *http.Response) map[string]string {
	if resp == nil || resp.Header == nil {
		return nil
	}
	headers := make(map[string]string, len(resp.Header))
	for key, values := range resp.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	return headers
}
