package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
)

// SSETransport implements the MCP SSE transport — server streams events
// from a long-lived GET, and the client POSTs JSON-RPC requests to a
// per-session "endpoint" URL the server emits on connect. Mirrors
// references/ai-sdk/packages/mcp/src/tool/mcp-sse-transport.ts.
type SSETransport struct {
	url          string
	headers      map[string]string
	authProvider OAuthClientProvider

	client      *http.Client
	messageURL  string
	eventSource io.ReadCloser

	requestID     atomic.Int64
	pending       map[int64]chan json.RawMessage
	pendingMu     sync.Mutex
	notifyHandler func(method string, params json.RawMessage)

	// endpointCh receives the resolved messageURL once the server emits the
	// first `endpoint` event. Closed (with error) if the stream ends before
	// the event arrives.
	endpointCh chan endpointResult

	closed bool
	mu     sync.Mutex
}

type endpointResult struct {
	url string
	err error
}

// NewSSETransport creates a new SSE transport. authProvider is optional;
// pass nil to skip OAuth.
func NewSSETransport(serverURL string, headers map[string]string, authProvider OAuthClientProvider) *SSETransport {
	return &SSETransport{
		url:          serverURL,
		headers:      headers,
		authProvider: authProvider,
		client:       &http.Client{},
		pending:      make(map[int64]chan json.RawMessage),
		endpointCh:   make(chan endpointResult, 1),
	}
}

// commonHeaders builds the standard MCP request headers — protocol
// version, optional bearer token from the auth provider, and the goai-mcp
// User-Agent suffix. Caller's `extra` overrides the per-request fields
// (Accept, Content-Type, …).
func (t *SSETransport) commonHeaders(ctx context.Context, extra http.Header) (http.Header, error) {
	h := http.Header{}
	for k, v := range t.headers {
		h.Set(k, v)
	}
	for k, vs := range extra {
		h[k] = vs
	}
	h.Set(HeaderProtocolVersion, LatestProtocolVersion)

	if t.authProvider != nil {
		tokens, err := t.authProvider.Tokens(ctx)
		if err != nil {
			return nil, fmt.Errorf("oauth tokens: %w", err)
		}
		if tokens != nil && tokens.AccessToken != "" {
			h.Set("Authorization", "Bearer "+tokens.AccessToken)
		}
	}

	prev := h.Get("User-Agent")
	if prev == "" {
		h.Set("User-Agent", UserAgentSuffix)
	} else if !strings.Contains(prev, UserAgentSuffix) {
		h.Set("User-Agent", prev+" "+UserAgentSuffix)
	}
	return h, nil
}

// Connect opens the SSE GET stream and waits for the server's first
// `endpoint` event before returning. Per the MCP SSE spec, the endpoint
// event's data is the URL to POST messages to (relative or absolute, but
// must be same-origin).
func (t *SSETransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	if t.eventSource != nil {
		t.mu.Unlock()
		return errors.New("already connected")
	}
	t.mu.Unlock()

	resp, err := t.openEventStream(ctx)
	if err != nil {
		return err
	}

	t.mu.Lock()
	t.eventSource = resp.Body
	t.mu.Unlock()

	go t.readLoop()

	select {
	case <-ctx.Done():
		t.eventSource.Close()
		return ctx.Err()
	case r := <-t.endpointCh:
		if r.err != nil {
			t.eventSource.Close()
			return r.err
		}
		t.mu.Lock()
		t.messageURL = r.url
		t.mu.Unlock()
		return nil
	}
}

// openEventStream issues the initial GET, transparently handling a single
// 401 → Auth → retry cycle when an authProvider is configured.
func (t *SSETransport) openEventStream(ctx context.Context) (*http.Response, error) {
	resp, err := t.doEventStream(ctx)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusUnauthorized && t.authProvider != nil {
		resourceMetaURL := ExtractResourceMetadataURL(resp)
		resp.Body.Close()
		res, err := Auth(ctx, t.authProvider, AuthOptions{
			ServerURL:           t.url,
			ResourceMetadataURL: resourceMetaURL,
		})
		if err != nil {
			return nil, err
		}
		if res != AuthResultAuthorized {
			return nil, &UnauthorizedError{Message: "authorization redirect required"}
		}
		resp, err = t.doEventStream(ctx)
		if err != nil {
			return nil, err
		}
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("MCP SSE Transport Error: %d %s: %s", resp.StatusCode, http.StatusText(resp.StatusCode), string(body))
	}
	return resp, nil
}

func (t *SSETransport) doEventStream(ctx context.Context) (*http.Response, error) {
	headers, err := t.commonHeaders(ctx, http.Header{"Accept": []string{"text/event-stream"}})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "GET", t.url, nil)
	if err != nil {
		return nil, fmt.Errorf("build SSE request: %w", err)
	}
	for k, vs := range headers {
		req.Header[k] = vs
	}
	return t.client.Do(req)
}

// resolveEndpoint validates the server-provided endpoint URL is same-origin
// and resolves it relative to the server URL.
func (t *SSETransport) resolveEndpoint(raw string) (string, error) {
	serverURL, err := url.Parse(t.url)
	if err != nil {
		return "", fmt.Errorf("parse server url: %w", err)
	}
	endpointURL, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse endpoint %q: %w", raw, err)
	}
	resolved := serverURL.ResolveReference(endpointURL)
	if resolved.Scheme+"://"+resolved.Host != serverURL.Scheme+"://"+serverURL.Host {
		return "", fmt.Errorf("MCP SSE Transport Error: endpoint origin %s does not match server %s", resolved.Host, serverURL.Host)
	}
	return resolved.String(), nil
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

	t.pendingMu.Lock()
	for _, ch := range t.pending {
		close(ch)
	}
	t.pending = make(map[int64]chan json.RawMessage)
	t.pendingMu.Unlock()

	return nil
}

// Send sends a JSON-RPC request via HTTP POST to the per-session endpoint
// returned during connect, awaiting the response on the SSE stream.
func (t *SSETransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, errors.New("transport closed")
	}
	messageURL := t.messageURL
	t.mu.Unlock()

	isNotification := strings.HasPrefix(method, "notifications/")

	if isNotification {
		reqBytes, err := json.Marshal(jsonRPCNotification{
			JSONRPC: "2.0",
			Method:  method,
			Params:  params,
		})
		if err != nil {
			return nil, fmt.Errorf("marshal notification: %w", err)
		}
		if err := t.postWithAuthRetry(ctx, messageURL, reqBytes); err != nil {
			return nil, err
		}
		return nil, nil
	}

	id := t.requestID.Add(1)

	respCh := make(chan json.RawMessage, 1)
	t.pendingMu.Lock()
	t.pending[id] = respCh
	t.pendingMu.Unlock()

	defer func() {
		t.pendingMu.Lock()
		delete(t.pending, id)
		t.pendingMu.Unlock()
	}()

	reqBytes, err := json.Marshal(jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if err := t.postWithAuthRetry(ctx, messageURL, reqBytes); err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case result, ok := <-respCh:
		if !ok {
			return nil, errors.New("connection closed")
		}
		return result, nil
	}
}

// postWithAuthRetry POSTs reqBytes to messageURL with one auth-on-401 retry.
func (t *SSETransport) postWithAuthRetry(ctx context.Context, messageURL string, reqBytes []byte) error {
	resp, err := t.doPost(ctx, messageURL, reqBytes)
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusUnauthorized && t.authProvider != nil {
		resourceMetaURL := ExtractResourceMetadataURL(resp)
		resp.Body.Close()
		res, err := Auth(ctx, t.authProvider, AuthOptions{
			ServerURL:           t.url,
			ResourceMetadataURL: resourceMetaURL,
		})
		if err != nil {
			return err
		}
		if res != AuthResultAuthorized {
			return &UnauthorizedError{Message: "authorization redirect required"}
		}
		resp, err = t.doPost(ctx, messageURL, reqBytes)
		if err != nil {
			return err
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("MCP SSE Transport Error: POSTing to endpoint (HTTP %d): %s", resp.StatusCode, string(body))
	}
	return nil
}

func (t *SSETransport) doPost(ctx context.Context, messageURL string, reqBytes []byte) (*http.Response, error) {
	headers, err := t.commonHeaders(ctx, http.Header{"Content-Type": []string{"application/json"}})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", messageURL, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	for k, vs := range headers {
		req.Header[k] = vs
	}
	return t.client.Do(req)
}

// OnNotification registers a handler for server-pushed notifications.
func (t *SSETransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}

func (t *SSETransport) readLoop() {
	endpointResolved := false
	err := scanSSE(t.eventSource, func(eventType, data, _ string) {
		if !endpointResolved {
			if eventType == "endpoint" {
				resolved, err := t.resolveEndpoint(data)
				select {
				case t.endpointCh <- endpointResult{url: resolved, err: err}:
				default:
				}
				endpointResolved = err == nil
				return
			}
			// First non-endpoint event is a protocol violation.
			select {
			case t.endpointCh <- endpointResult{err: fmt.Errorf("MCP SSE Transport Error: expected first event to be 'endpoint', got %q", eventType)}:
			default:
			}
			return
		}
		t.handleEvent(eventType, data)
	})
	if !endpointResolved {
		errMsg := errors.New("MCP SSE Transport Error: stream closed before endpoint event")
		if err != nil {
			errMsg = err
		}
		select {
		case t.endpointCh <- endpointResult{err: errMsg}:
		default:
		}
	}
}

func (t *SSETransport) handleEvent(eventType, data string) {
	if eventType != "" && eventType != "message" {
		return
	}
	var msg jsonRPCMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		return
	}

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

	if t.notifyHandler != nil && msg.Method != "" {
		t.notifyHandler(msg.Method, msg.Params)
	}
}
