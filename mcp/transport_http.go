package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
)

// HTTPTransport implements the Transport interface using HTTP.
// This is a simpler transport that uses HTTP POST for request/response.
type HTTPTransport struct {
	url     string
	headers map[string]string

	client    *http.Client
	requestID atomic.Int64

	closed bool
	mu     sync.Mutex

	notifyHandler func(method string, params json.RawMessage)
}

// NewHTTPTransport creates a new HTTP transport.
func NewHTTPTransport(url string, headers map[string]string) *HTTPTransport {
	return &HTTPTransport{
		url:     url,
		headers: headers,
		client:  &http.Client{},
	}
}

// Connect initializes the HTTP transport.
// For HTTP transport, this is a no-op since connections are made per-request.
func (t *HTTPTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return fmt.Errorf("transport closed")
	}

	return nil
}

// Close closes the HTTP transport.
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.closed = true
	return nil
}

// Send sends a JSON-RPC request via HTTP POST.
// Notifications (methods starting with "notifications/") are sent without an id
// and the response body is not parsed (server responds with 202 or 200).
func (t *HTTPTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, fmt.Errorf("transport closed")
	}
	t.mu.Unlock()

	isNotification := strings.HasPrefix(method, "notifications/")

	// Build JSON-RPC message: notifications have no id
	var reqBytes []byte
	var err error
	if isNotification {
		reqBytes, err = json.Marshal(jsonRPCNotification{
			JSONRPC: "2.0",
			Method:  method,
			Params:  params,
		})
	} else {
		id := t.requestID.Add(1)
		reqBytes, err = json.Marshal(jsonRPCRequest{
			JSONRPC: "2.0",
			ID:      id,
			Method:  method,
			Params:  params,
		})
	}
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send HTTP POST
	req, err := http.NewRequestWithContext(ctx, "POST", t.url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Notifications: accept 2xx, ignore body (server may return 202 or 200)
	if isNotification {
		io.Copy(io.Discard, resp.Body)
		return nil, nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var msg jsonRPCMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if msg.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", msg.Error.Code, msg.Error.Message)
	}

	return msg.Result, nil
}

// OnNotification registers a handler for server notifications.
// Note: HTTP transport doesn't support server-initiated notifications.
func (t *HTTPTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}
