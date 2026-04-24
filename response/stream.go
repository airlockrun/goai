package response

import (
	"bufio"
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/schema"
)

// CreateEventSourceResponseHandler creates a handler for SSE (Server-Sent Events) streams.
// It parses each SSE data line as JSON and sends the results to a channel.
// The channel is closed when the stream ends or encounters an error.
func CreateEventSourceResponseHandler[T any](s *schema.Schema) Handler[<-chan *ParseResult[T]] {
	return func(opts HandlerOptions) (*HandlerResult[<-chan *ParseResult[T]], error) {
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

		ch := make(chan *ParseResult[T], 100)

		go func() {
			defer close(ch)
			defer opts.Response.Body.Close()

			scanner := bufio.NewScanner(opts.Response.Body)
			scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

			for scanner.Scan() {
				line := scanner.Text()

				// Parse SSE data lines
				if !strings.HasPrefix(line, "data:") {
					continue
				}

				data := strings.TrimPrefix(line, "data:")
				if len(data) > 0 && data[0] == ' ' {
					data = data[1:]
				}

				// Skip [DONE] marker (OpenAI convention)
				if data == "[DONE]" {
					continue
				}

				// Parse JSON
				result := parseSSEData[T](data, s)
				ch <- result
			}
		}()

		return &HandlerResult[<-chan *ParseResult[T]]{
			Value:           ch,
			ResponseHeaders: responseHeaders,
		}, nil
	}
}

// parseSSEData parses SSE data as JSON with optional schema validation.
func parseSSEData[T any](data string, _ *schema.Schema) *ParseResult[T] {
	// Parse raw JSON first
	var rawValue any
	if err := json.Unmarshal([]byte(data), &rawValue); err != nil {
		return &ParseResult[T]{
			Success: false,
			Error:   &errors.JSONParseError{Text: data, Cause: err},
		}
	}

	// Parse into typed value
	var value T
	if err := json.Unmarshal([]byte(data), &value); err != nil {
		return &ParseResult[T]{
			Success:  false,
			Error:    &errors.JSONParseError{Text: data, Cause: err},
			RawValue: rawValue,
		}
	}

	return &ParseResult[T]{
		Success:  true,
		Value:    value,
		RawValue: rawValue,
	}
}
