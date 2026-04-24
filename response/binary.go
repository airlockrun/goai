package response

import (
	"io"

	"github.com/airlockrun/goai/errors"
)

// CreateBinaryResponseHandler creates a handler for binary responses.
// It reads the entire response body and returns it as a byte slice.
// Returns an error if the response body is empty or cannot be read.
func CreateBinaryResponseHandler() Handler[[]byte] {
	return func(opts HandlerOptions) (*HandlerResult[[]byte], error) {
		responseHeaders := ExtractResponseHeaders(opts.Response)

		if opts.Response.Body == nil {
			return nil, errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           "Response body is empty",
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
			})
		}

		data, err := io.ReadAll(opts.Response.Body)
		if err != nil {
			return nil, errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           "Failed to read response as array buffer",
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
				Cause:             err,
			})
		}

		if len(data) == 0 {
			return nil, errors.NewAPICallError(errors.APICallErrorOptions{
				Message:           "Response body is empty",
				URL:               opts.URL,
				RequestBodyValues: opts.RequestBodyValues,
				StatusCode:        opts.Response.StatusCode,
				ResponseHeaders:   responseHeaders,
			})
		}

		return &HandlerResult[[]byte]{
			Value:           data,
			ResponseHeaders: responseHeaders,
		}, nil
	}
}
