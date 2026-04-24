// Package internal provides shared utilities for the goai package.
package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient wraps http.Client with goai-specific functionality.
type HTTPClient struct {
	// Client is the underlying HTTP client.
	Client *http.Client

	// BaseURL is the base URL for all requests.
	BaseURL string

	// DefaultHeaders are headers sent with every request.
	DefaultHeaders map[string]string

	// MaxRetries is the maximum number of retry attempts.
	MaxRetries int

	// RetryBackoff is the initial backoff duration for retries.
	RetryBackoff time.Duration
}

// NewHTTPClient creates a new HTTP client with sensible defaults.
func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		Client: &http.Client{
			Timeout: 120 * time.Second,
		},
		BaseURL:        baseURL,
		DefaultHeaders: make(map[string]string),
		MaxRetries:     3,
		RetryBackoff:   1 * time.Second,
	}
}

// RequestOptions contains options for an HTTP request.
type RequestOptions struct {
	// Method is the HTTP method (GET, POST, etc.).
	Method string

	// Path is the URL path (appended to BaseURL).
	Path string

	// Headers are additional headers for this request.
	Headers map[string]string

	// Body is the request body (will be JSON-encoded if not nil).
	Body any

	// RawBody is the raw request body (used instead of Body if set).
	RawBody io.Reader

	// ContentType is the Content-Type header (defaults to "application/json").
	ContentType string

	// Stream indicates this is a streaming request.
	Stream bool
}

// Response contains the HTTP response.
type Response struct {
	// StatusCode is the HTTP status code.
	StatusCode int

	// Headers are the response headers.
	Headers http.Header

	// Body is the response body.
	Body []byte

	// Reader provides streaming access to the response body.
	Reader io.ReadCloser
}

// JSON decodes the response body as JSON into v.
func (r *Response) JSON(v any) error {
	return json.Unmarshal(r.Body, v)
}

// Do executes an HTTP request.
func (c *HTTPClient) Do(ctx context.Context, opts RequestOptions) (*Response, error) {
	url := c.BaseURL + opts.Path

	var bodyReader io.Reader
	if opts.RawBody != nil {
		bodyReader = opts.RawBody
	} else if opts.Body != nil {
		bodyBytes, err := json.Marshal(opts.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, opts.Method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set default headers
	for k, v := range c.DefaultHeaders {
		req.Header.Set(k, v)
	}

	// Set request-specific headers
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	// Set content type
	if opts.ContentType != "" {
		req.Header.Set("Content-Type", opts.ContentType)
	} else if opts.Body != nil || opts.RawBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	result := &Response{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
	}

	if opts.Stream {
		result.Reader = resp.Body
	} else {
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}
		result.Body = body
	}

	return result, nil
}

// Get performs a GET request.
func (c *HTTPClient) Get(ctx context.Context, path string, headers map[string]string) (*Response, error) {
	return c.Do(ctx, RequestOptions{
		Method:  "GET",
		Path:    path,
		Headers: headers,
	})
}

// Post performs a POST request with JSON body.
func (c *HTTPClient) Post(ctx context.Context, path string, body any, headers map[string]string) (*Response, error) {
	return c.Do(ctx, RequestOptions{
		Method:  "POST",
		Path:    path,
		Body:    body,
		Headers: headers,
	})
}

// PostStream performs a POST request and returns a streaming response.
func (c *HTTPClient) PostStream(ctx context.Context, path string, body any, headers map[string]string) (*Response, error) {
	return c.Do(ctx, RequestOptions{
		Method:  "POST",
		Path:    path,
		Body:    body,
		Headers: headers,
		Stream:  true,
	})
}

// SetAuthBearer sets the Authorization header with a Bearer token.
func (c *HTTPClient) SetAuthBearer(token string) {
	c.DefaultHeaders["Authorization"] = "Bearer " + token
}

// SetAuthAPIKey sets an API key header.
func (c *HTTPClient) SetAuthAPIKey(headerName, apiKey string) {
	c.DefaultHeaders[headerName] = apiKey
}

// MultipartWriter helps build multipart/form-data requests.
type MultipartWriter struct {
	buf         *bytes.Buffer
	boundary    string
	contentType string
}

// NewMultipartWriter creates a new multipart writer.
func NewMultipartWriter() *MultipartWriter {
	boundary := fmt.Sprintf("----GoAI%d", time.Now().UnixNano())
	return &MultipartWriter{
		buf:         &bytes.Buffer{},
		boundary:    boundary,
		contentType: "multipart/form-data; boundary=" + boundary,
	}
}

// WriteField writes a text field.
func (w *MultipartWriter) WriteField(name, value string) {
	fmt.Fprintf(w.buf, "--%s\r\n", w.boundary)
	fmt.Fprintf(w.buf, "Content-Disposition: form-data; name=\"%s\"\r\n\r\n", name)
	fmt.Fprintf(w.buf, "%s\r\n", value)
}

// WriteFile writes a file field.
func (w *MultipartWriter) WriteFile(name, filename, contentType string, data []byte) {
	fmt.Fprintf(w.buf, "--%s\r\n", w.boundary)
	fmt.Fprintf(w.buf, "Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n", name, filename)
	fmt.Fprintf(w.buf, "Content-Type: %s\r\n\r\n", contentType)
	w.buf.Write(data)
	fmt.Fprintf(w.buf, "\r\n")
}

// Close finalizes the multipart body.
func (w *MultipartWriter) Close() {
	fmt.Fprintf(w.buf, "--%s--\r\n", w.boundary)
}

// ContentType returns the Content-Type header value.
func (w *MultipartWriter) ContentType() string {
	return w.contentType
}

// Reader returns the multipart body reader.
func (w *MultipartWriter) Reader() io.Reader {
	return w.buf
}

// Bytes returns the multipart body as bytes.
func (w *MultipartWriter) Bytes() []byte {
	return w.buf.Bytes()
}
