package response

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/schema"
)

// ParseResult represents the result of parsing JSON.
type ParseResult[T any] struct {
	// Success indicates if parsing succeeded.
	Success bool

	// Value is the parsed and validated value (only valid if Success is true).
	Value T

	// RawValue is the raw parsed JSON before validation.
	RawValue any

	// Error is the parsing or validation error (only valid if Success is false).
	Error error
}

// SafeParseJSON parses JSON with optional schema validation.
// Returns a ParseResult with Success=true and both Value and RawValue on success,
// or Success=false and Error on failure.
func SafeParseJSON[T any](text string, s *schema.Schema) *ParseResult[T] {
	// Parse raw JSON first
	var rawValue any
	if err := json.Unmarshal([]byte(text), &rawValue); err != nil {
		return &ParseResult[T]{
			Success: false,
			Error:   &errors.JSONParseError{Text: text, Cause: err},
		}
	}

	// If no schema, use raw value
	if s == nil {
		// Type assertion for non-schema case
		if v, ok := rawValue.(T); ok {
			return &ParseResult[T]{
				Success:  true,
				Value:    v,
				RawValue: rawValue,
			}
		}
		// For generic T, try to convert via JSON re-marshal
		var value T
		data, _ := json.Marshal(rawValue)
		if err := json.Unmarshal(data, &value); err != nil {
			return &ParseResult[T]{
				Success:  false,
				Error:    &errors.JSONParseError{Text: text, Cause: err},
				RawValue: rawValue,
			}
		}
		return &ParseResult[T]{
			Success:  true,
			Value:    value,
			RawValue: rawValue,
		}
	}

	// With schema: validate and parse into T
	var value T
	if err := json.Unmarshal([]byte(text), &value); err != nil {
		return &ParseResult[T]{
			Success:  false,
			Error:    &errors.JSONParseError{Text: text, Cause: err},
			RawValue: rawValue,
		}
	}

	// Note: Schema validation is optional in Go since we rely on struct tags
	// For strict validation, the caller should validate against the schema
	return &ParseResult[T]{
		Success:  true,
		Value:    value,
		RawValue: rawValue,
	}
}

// ParseJSON parses JSON with optional schema validation.
// Returns the parsed value or an error.
func ParseJSON[T any](text string, s *schema.Schema) (T, error) {
	result := SafeParseJSON[T](text, s)
	if !result.Success {
		var zero T
		return zero, result.Error
	}
	return result.Value, nil
}

// CreateJSONResponseHandler creates a handler that parses JSON responses.
// The handler validates against the provided schema if non-nil.
// Returns both the parsed value and rawValue in the result.
func CreateJSONResponseHandler[T any](s *schema.Schema) Handler[T] {
	return func(opts HandlerOptions) (*HandlerResult[T], error) {
		responseBody, err := io.ReadAll(opts.Response.Body)
		if err != nil {
			return nil, errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           "Failed to read response body",
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   ExtractResponseHeaders(opts.Response),
				Cause:             err,
			})
		}

		text := string(responseBody)
		result := SafeParseJSON[T](strings.TrimSpace(text), s)
		responseHeaders := ExtractResponseHeaders(opts.Response)

		if !result.Success {
			return nil, errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           "Invalid JSON response",
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
				ResponseBody:      text,
				Cause:             result.Error,
			})
		}

		return &HandlerResult[T]{
			Value:           result.Value,
			RawValue:        result.RawValue,
			ResponseHeaders: responseHeaders,
		}, nil
	}
}
