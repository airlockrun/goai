package response

import (
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/errors"
)

// JSONErrorConfig configures the JSON error response handler.
type JSONErrorConfig[T any] struct {
	// ErrorSchema parses the error JSON body into the provider-specific error type.
	// Returns the parsed error and any parsing error.
	ErrorSchema func(body []byte) (T, error)

	// ErrorToMessage extracts a user-facing message from the parsed error.
	ErrorToMessage func(error T) string

	// IsRetryable determines if the error is retryable based on response and parsed error.
	// If nil, default retryability logic is used based on status code.
	IsRetryable func(resp *http.Response, error *T) bool
}

// CreateJSONErrorResponseHandler creates a handler for JSON error responses.
// It attempts to parse the response body as JSON using the provided schema.
// Falls back to status text if the body is empty or JSON parsing fails.
func CreateJSONErrorResponseHandler[T any](config JSONErrorConfig[T]) Handler[*errors.APICallError] {
	return func(opts HandlerOptions) (*HandlerResult[*errors.APICallError], error) {
		responseBody, err := io.ReadAll(opts.Response.Body)
		if err != nil {
			responseBody = nil
		}

		responseBodyStr := string(responseBody)
		responseHeaders := ExtractResponseHeaders(opts.Response)

		// Handle empty response body
		if strings.TrimSpace(responseBodyStr) == "" {
			var isRetryable bool
			if config.IsRetryable != nil {
				isRetryable = config.IsRetryable(opts.Response, nil)
			}
			return &HandlerResult[*errors.APICallError]{
				ResponseHeaders: responseHeaders,
				Value: errors.NewAPICallError(errors.APICallErrorOptions{
					Message:           opts.Response.Status,
					URL:               opts.URL,
					RequestBodyValues: opts.RequestBodyValues,
					StatusCode:        opts.Response.StatusCode,
					ResponseHeaders:   responseHeaders,
					ResponseBody:      responseBodyStr,
					IsRetryable:       isRetryable,
					IsRetryableSet:    config.IsRetryable != nil,
				}),
			}, nil
		}

		// Try to parse the error JSON
		parsedError, parseErr := config.ErrorSchema(responseBody)
		if parseErr != nil {
			// Failed to parse JSON - fall back to status text
			var isRetryable bool
			if config.IsRetryable != nil {
				isRetryable = config.IsRetryable(opts.Response, nil)
			}
			return &HandlerResult[*errors.APICallError]{
				ResponseHeaders: responseHeaders,
				Value: errors.NewAPICallError(errors.APICallErrorOptions{
					Message:           opts.Response.Status,
					URL:               opts.URL,
					RequestBodyValues: opts.RequestBodyValues,
					StatusCode:        opts.Response.StatusCode,
					ResponseHeaders:   responseHeaders,
					ResponseBody:      responseBodyStr,
					IsRetryable:       isRetryable,
					IsRetryableSet:    config.IsRetryable != nil,
				}),
			}, nil
		}

		// Successfully parsed error
		var isRetryable bool
		if config.IsRetryable != nil {
			isRetryable = config.IsRetryable(opts.Response, &parsedError)
		}

		return &HandlerResult[*errors.APICallError]{
			ResponseHeaders: responseHeaders,
			Value: errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           config.ErrorToMessage(parsedError),
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
				ResponseBody:      responseBodyStr,
				Data:              parsedError,
				IsRetryable:       isRetryable,
				IsRetryableSet:    config.IsRetryable != nil,
			}),
		}, nil
	}
}

// CreateStatusCodeErrorResponseHandler creates a generic error handler.
// It uses the HTTP status text as the error message and includes the response body.
func CreateStatusCodeErrorResponseHandler() Handler[*errors.APICallError] {
	return func(opts HandlerOptions) (*HandlerResult[*errors.APICallError], error) {
		responseBody, err := io.ReadAll(opts.Response.Body)
		if err != nil {
			responseBody = nil
		}

		responseHeaders := ExtractResponseHeaders(opts.Response)

		return &HandlerResult[*errors.APICallError]{
			ResponseHeaders: responseHeaders,
			Value: errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           opts.Response.Status,
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
				ResponseBody:      string(responseBody),
			}),
		}, nil
	}
}
