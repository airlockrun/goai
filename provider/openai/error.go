package openai

import (
	"encoding/json"
	"net/http"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/response"
)

// OpenAIErrorData represents the structure of an OpenAI API error response.
// This schema is designed to handle both standard OpenAI errors and
// wrapped errors from OpenAI-compatible providers (e.g., OpenRouter).
//
// Source: ai-sdk/packages/openai/src/openai-error.ts
type OpenAIErrorData struct {
	Error OpenAIErrorInfo `json:"error"`
}

// OpenAIErrorInfo contains the details of an OpenAI error.
type OpenAIErrorInfo struct {
	// Message is the error message. For nested errors from providers like
	// OpenRouter, this may contain JSON-encoded error details.
	Message string `json:"message"`

	// Type is the error type (e.g., "invalid_request_error").
	// Optional - may be nil for some providers.
	Type *string `json:"type,omitempty"`

	// Param is the parameter that caused the error.
	// Optional - type is any to handle different provider formats.
	Param any `json:"param,omitempty"`

	// Code is the error code. Can be either a string or number depending
	// on the provider. For example, OpenAI uses strings like "invalid_api_key",
	// while OpenRouter may use numeric codes like 429.
	Code any `json:"code,omitempty"`
}

// ParseOpenAIError parses an OpenAI API error response.
// Returns the parsed error data or an error if parsing fails.
func ParseOpenAIError(body []byte) (*OpenAIErrorData, error) {
	var data OpenAIErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}
	return &data, nil
}

// GetMessage returns the error message.
func (e *OpenAIErrorData) GetMessage() string {
	return e.Error.Message
}

// GetType returns the error type, or empty string if not set.
func (e *OpenAIErrorData) GetType() string {
	if e.Error.Type == nil {
		return ""
	}
	return *e.Error.Type
}

// GetCodeString returns the error code as a string.
// If the code is numeric, it returns an empty string.
func (e *OpenAIErrorData) GetCodeString() string {
	if code, ok := e.Error.Code.(string); ok {
		return code
	}
	return ""
}

// GetCodeNumber returns the error code as a number.
// If the code is not numeric, it returns 0.
func (e *OpenAIErrorData) GetCodeNumber() int {
	switch code := e.Error.Code.(type) {
	case float64:
		return int(code)
	case int:
		return code
	case int64:
		return int(code)
	default:
		return 0
	}
}

// parseOpenAIErrorForHandler wraps ParseOpenAIError for use with response handlers.
func parseOpenAIErrorForHandler(body []byte) (OpenAIErrorData, error) {
	data, err := ParseOpenAIError(body)
	if err != nil {
		return OpenAIErrorData{}, err
	}
	return *data, nil
}

// OpenAIFailedResponseHandler is the error response handler for OpenAI API errors.
// It parses the JSON error response and extracts the error message.
// This matches ai-sdk's openaiFailedResponseHandler.
//
// Source: ai-sdk/packages/openai/src/openai-error.ts
var OpenAIFailedResponseHandler = response.CreateJSONErrorResponseHandler(response.JSONErrorConfig[OpenAIErrorData]{
	ErrorSchema:    parseOpenAIErrorForHandler,
	ErrorToMessage: func(data OpenAIErrorData) string { return data.GetMessage() },
	IsRetryable: func(resp *http.Response, _ *OpenAIErrorData) bool {
		return resp.StatusCode == 429 || resp.StatusCode >= 500
	},
})

// HandleErrorResponse processes an error response using the OpenAI error handler.
// Returns an APICallError with parsed error data.
func HandleErrorResponse(resp *http.Response, url string, requestBodyValues any) *errors.APICallError {
	result, err := OpenAIFailedResponseHandler(response.HandlerOptions{
		URL:               url,
		RequestBodyValues: requestBodyValues,
		Response:          resp,
	})
	if err != nil {
		// If handler itself fails, return a generic error
		return errors.NewAPICallError(errors.APICallErrorOptions{
			Message:           "Failed to process error response",
			URL:               url,
			RequestBodyValues: requestBodyValues,
			StatusCode:        resp.StatusCode,
			Cause:             err,
		})
	}
	return result.Value
}
