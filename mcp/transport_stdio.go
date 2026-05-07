package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
)

// StdioTransport implements the Transport interface using stdio.
type StdioTransport struct {
	command string
	args    []string
	env     map[string]string

	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  io.ReadCloser
	scanner *bufio.Scanner

	requestID     atomic.Int64
	pending       map[int64]chan json.RawMessage
	pendingMu     sync.Mutex
	notifyHandler func(method string, params json.RawMessage)

	closed bool
	mu     sync.Mutex
}

// NewStdioTransport creates a new stdio transport.
func NewStdioTransport(command string, args []string, env map[string]string) *StdioTransport {
	return &StdioTransport{
		command: command,
		args:    args,
		env:     env,
		pending: make(map[int64]chan json.RawMessage),
	}
}

// SetProtocolVersion is a no-op for stdio — there's no HTTP header to set.
// Implements Transport.
func (t *StdioTransport) SetProtocolVersion(string) {}

// Connect starts the subprocess and establishes communication.
func (t *StdioTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cmd != nil {
		return fmt.Errorf("already connected")
	}

	t.cmd = exec.CommandContext(ctx, t.command, t.args...)

	// Set environment
	if len(t.env) > 0 {
		for k, v := range t.env {
			t.cmd.Env = append(t.cmd.Env, fmt.Sprintf("%s=%s", k, v))
		}
	}

	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	t.stdout, err = t.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	if err := t.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start command: %w", err)
	}

	t.scanner = bufio.NewScanner(t.stdout)
	t.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Start reading responses
	go t.readLoop()

	return nil
}

// Close terminates the subprocess.
func (t *StdioTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}
	t.closed = true

	if t.stdin != nil {
		t.stdin.Close()
	}

	if t.cmd != nil && t.cmd.Process != nil {
		t.cmd.Process.Kill()
		t.cmd.Wait()
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

// Send sends a JSON-RPC request and waits for the response.
// Notifications (methods starting with "notifications/") are fire-and-forget:
// sent without an id and no response is expected.
func (t *StdioTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, fmt.Errorf("transport closed")
	}
	t.mu.Unlock()

	// Notifications: no id, no response expected
	if strings.HasPrefix(method, "notifications/") {
		notif := jsonRPCNotification{
			JSONRPC: "2.0",
			Method:  method,
			Params:  params,
		}
		notifBytes, err := json.Marshal(notif)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal notification: %w", err)
		}
		t.mu.Lock()
		_, err = fmt.Fprintf(t.stdin, "%s\n", notifBytes)
		t.mu.Unlock()
		if err != nil {
			return nil, fmt.Errorf("failed to send notification: %w", err)
		}
		return nil, nil
	}

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
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request
	t.mu.Lock()
	_, err = fmt.Fprintf(t.stdin, "%s\n", reqBytes)
	t.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Wait for response
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
func (t *StdioTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}

func (t *StdioTransport) readLoop() {
	for t.scanner.Scan() {
		line := t.scanner.Text()
		if line == "" {
			continue
		}

		var msg jsonRPCMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			continue
		}

		// Check if it's a response (has ID)
		if msg.ID != nil {
			t.pendingMu.Lock()
			if ch, ok := t.pending[*msg.ID]; ok {
				if msg.Error != nil {
					// Send error as nil result - caller will need to handle
					ch <- nil
				} else {
					ch <- msg.Result
				}
			}
			t.pendingMu.Unlock()
			continue
		}

		// It's a notification
		if t.notifyHandler != nil && msg.Method != "" {
			t.notifyHandler(msg.Method, msg.Params)
		}
	}
}

// JSON-RPC types

type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// jsonRPCNotification is a JSON-RPC notification (no id field, no response expected).
type jsonRPCNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonRPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}
