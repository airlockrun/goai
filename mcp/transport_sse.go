package mcp

import (
	"bufio"
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

// SSETransport implements the Transport interface using Server-Sent Events.
type SSETransport struct {
	url     string
	headers map[string]string

	client       *http.Client
	messageURL   string
	eventSource  io.ReadCloser
	eventScanner *bufio.Scanner

	requestID     atomic.Int64
	pending       map[int64]chan json.RawMessage
	pendingMu     sync.Mutex
	notifyHandler func(method string, params json.RawMessage)

	closed bool
	mu     sync.Mutex
}

// NewSSETransport creates a new SSE transport.
func NewSSETransport(url string, headers map[string]string) *SSETransport {
	return &SSETransport{
		url:     url,
		headers: headers,
		client:  &http.Client{},
		pending: make(map[int64]chan json.RawMessage),
	}
}

// Connect establishes the SSE connection.
func (t *SSETransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.eventSource != nil {
		return fmt.Errorf("already connected")
	}

	// Make initial request to get the SSE endpoint
	req, err := http.NewRequestWithContext(ctx, "GET", t.url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Accept", "text/event-stream")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	t.eventSource = resp.Body
	t.eventScanner = bufio.NewScanner(t.eventSource)
	t.eventScanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// The message URL is typically in the response headers or first event
	t.messageURL = t.url + "/message"

	// Start reading events
	go t.readLoop()

	return nil
}

// Close closes the SSE connection.
func (t *SSETransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}
	t.closed = true

	if t.eventSource != nil {
		t.eventSource.Close()
	}

	// Cancel all pending requests
	t.pendingMu.Lock()
	for _, ch := range t.pending {
		close(ch)
	}
	t.pending = make(map[int64]chan json.RawMessage)
	t.pendingMu.Unlock()

	return nil
}

// Send sends a JSON-RPC request via HTTP POST.
func (t *SSETransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, fmt.Errorf("transport closed")
	}
	messageURL := t.messageURL
	t.mu.Unlock()

	id := t.requestID.Add(1)

	// Create response channel
	respCh := make(chan json.RawMessage, 1)
	t.pendingMu.Lock()
	t.pending[id] = respCh
	t.pendingMu.Unlock()

	defer func() {
		t.pendingMu.Lock()
		delete(t.pending, id)
		t.pendingMu.Unlock()
	}()

	// Build request
	reqBody := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send HTTP POST
	req, err := http.NewRequestWithContext(ctx, "POST", messageURL, bytes.NewReader(reqBytes))
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

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed (status %d): %s", resp.StatusCode, string(body))
	}

	// Wait for response via SSE
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case result, ok := <-respCh:
		if !ok {
			return nil, fmt.Errorf("connection closed")
		}
		return result, nil
	}
}

// OnNotification registers a handler for server notifications.
func (t *SSETransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}

func (t *SSETransport) readLoop() {
	var eventType string
	var dataLines []string

	for t.eventScanner.Scan() {
		line := t.eventScanner.Text()

		if line == "" {
			// Empty line signals end of event
			if len(dataLines) > 0 {
				data := strings.Join(dataLines, "\n")
				t.handleEvent(eventType, data)
			}
			eventType = ""
			dataLines = nil
			continue
		}

		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

func (t *SSETransport) handleEvent(eventType, data string) {
	var msg jsonRPCMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return
	}

	// Check if it's a response (has ID)
	if msg.ID != nil {
		t.pendingMu.Lock()
		if ch, ok := t.pending[*msg.ID]; ok {
			if msg.Error != nil {
				ch <- nil
			} else {
				ch <- msg.Result
			}
		}
		t.pendingMu.Unlock()
		return
	}

	// It's a notification
	if t.notifyHandler != nil && msg.Method != "" {
		t.notifyHandler(msg.Method, msg.Params)
	}
}
