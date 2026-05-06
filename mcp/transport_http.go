package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// HTTPTransport implements the MCP Streamable HTTP transport. Each
// JSON-RPC call is a POST; the server may respond with a single
// application/json body, a JSON array, or an SSE stream. A long-lived
// inbound GET stream carries server-initiated notifications. Mirrors
// references/ai-sdk/packages/mcp/src/tool/mcp-http-transport.ts.
type HTTPTransport struct {
	url          string
	headers      map[string]string
	authProvider OAuthClientProvider

	client    *http.Client
	requestID atomic.Int64

	// sessionID stores the mcp-session-id the server set on initialize.
	// Atomic load/store so reads on the inbound goroutine don't race the
	// POST handler.
	sessionID atomic.Value // string

	notifyHandler func(method string, params json.RawMessage)
	notifyMu      sync.Mutex

	inboundCancel context.CancelFunc
	inboundWG     sync.WaitGroup

	closed bool
	mu     sync.Mutex
}

// NewHTTPTransport creates a new HTTP transport. authProvider is optional;
// pass nil to skip OAuth.
func NewHTTPTransport(serverURL string, headers map[string]string, authProvider OAuthClientProvider) *HTTPTransport {
	t := &HTTPTransport{
		url:          serverURL,
		headers:      headers,
		authProvider: authProvider,
		client:       &http.Client{},
	}
	t.sessionID.Store("")
	return t
}

// Connect spawns the optional inbound SSE goroutine. The MCP HTTP transport
// is otherwise per-request — no long-lived connection state up front.
func (t *HTTPTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return errors.New("transport closed")
	}
	if t.inboundCancel != nil {
		return errors.New("already connected")
	}

	inboundCtx, cancel := context.WithCancel(context.Background())
	t.inboundCancel = cancel
	t.inboundWG.Add(1)
	go func() {
		defer t.inboundWG.Done()
		t.runInboundSSE(inboundCtx, "")
	}()
	return nil
}

// Close terminates the session via DELETE (when a session-id is set),
// stops the inbound SSE goroutine, and closes the HTTP client.
func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil
	}
	t.closed = true
	cancel := t.inboundCancel
	t.inboundCancel = nil
	t.mu.Unlock()

	sid, _ := t.sessionID.Load().(string)
	if sid != "" {
		ctx, cancelDel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelDel()
		if headers, err := t.commonHeaders(ctx, nil); err == nil {
			req, err := http.NewRequestWithContext(ctx, "DELETE", t.url, nil)
			if err == nil {
				for k, vs := range headers {
					req.Header[k] = vs
				}
				resp, err := t.client.Do(req)
				if err == nil {
					resp.Body.Close()
				}
			}
		}
	}

	if cancel != nil {
		cancel()
	}
	t.inboundWG.Wait()
	return nil
}

// commonHeaders builds the standard MCP request headers.
func (t *HTTPTransport) commonHeaders(ctx context.Context, extra http.Header) (http.Header, error) {
	h := http.Header{}
	for k, v := range t.headers {
		h.Set(k, v)
	}
	for k, vs := range extra {
		h[k] = vs
	}
	h.Set(HeaderProtocolVersion, LatestProtocolVersion)

	if sid, _ := t.sessionID.Load().(string); sid != "" {
		h.Set(HeaderSessionID, sid)
	}

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

// Send posts a JSON-RPC request to the server. Notifications skip the
// response wait; non-notifications return either the inline result (JSON
// or SSE) or an error.
func (t *HTTPTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil, errors.New("transport closed")
	}
	t.mu.Unlock()

	isNotification := strings.HasPrefix(method, "notifications/")

	var body []byte
	var requestID int64
	if isNotification {
		b, err := json.Marshal(jsonRPCNotification{
			JSONRPC: "2.0",
			Method:  method,
			Params:  params,
		})
		if err != nil {
			return nil, fmt.Errorf("marshal notification: %w", err)
		}
		body = b
	} else {
		requestID = t.requestID.Add(1)
		b, err := json.Marshal(jsonRPCRequest{
			JSONRPC: "2.0",
			ID:      requestID,
			Method:  method,
			Params:  params,
		})
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}
		body = b
	}

	resp, err := t.postWithAuthRetry(ctx, body)
	if err != nil {
		return nil, err
	}
	// Capture session id from any response.
	if sid := resp.Header.Get(HeaderSessionID); sid != "" {
		t.sessionID.Store(sid)
	}
	defer resp.Body.Close()

	if isNotification {
		// Notifications — server may return 200 with ack body or 202.
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
			return nil, t.statusError(resp)
		}
		io.Copy(io.Discard, resp.Body)
		return nil, nil
	}

	if resp.StatusCode == http.StatusAccepted {
		// Server accepted but won't return an inline result. Caller would
		// need to wait for a server-pushed message on the inbound SSE; we
		// don't have a request-id correlation channel for that yet, so
		// return nil result. Callers expecting a result should require 200.
		io.Copy(io.Discard, resp.Body)
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, t.statusError(resp)
	}

	contentType := resp.Header.Get("Content-Type")
	switch {
	case strings.Contains(contentType, "application/json"):
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("read response: %w", err)
		}
		return t.parseJSONResponse(raw, requestID)
	case strings.Contains(contentType, "text/event-stream"):
		return t.parseSSEResponse(resp.Body, requestID)
	default:
		return nil, fmt.Errorf("MCP HTTP Transport Error: unexpected content type: %s", contentType)
	}
}

func (t *HTTPTransport) statusError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	msg := fmt.Sprintf("MCP HTTP Transport Error: POSTing to endpoint (HTTP %d): %s", resp.StatusCode, string(body))
	if resp.StatusCode == http.StatusNotFound {
		msg += ". This server does not support HTTP transport. Try using `sse` transport instead"
	}
	return errors.New(msg)
}

// postWithAuthRetry runs one POST plus, optionally, one auth-on-401 retry
// when an authProvider is configured.
func (t *HTTPTransport) postWithAuthRetry(ctx context.Context, body []byte) (*http.Response, error) {
	resp, err := t.doPost(ctx, body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusUnauthorized || t.authProvider == nil {
		return resp, nil
	}
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
	return t.doPost(ctx, body)
}

func (t *HTTPTransport) doPost(ctx context.Context, body []byte) (*http.Response, error) {
	headers, err := t.commonHeaders(ctx, http.Header{
		"Content-Type": []string{"application/json"},
		"Accept":       []string{"application/json, text/event-stream"},
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", t.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	for k, vs := range headers {
		req.Header[k] = vs
	}
	return t.client.Do(req)
}

// parseJSONResponse handles `application/json` responses. The server may
// reply with either a single JSON-RPC message or an array of them. We
// return the result for our request id and route the rest as
// notifications (when they have no id) — matching ai-sdk's behavior.
func (t *HTTPTransport) parseJSONResponse(raw []byte, requestID int64) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, errors.New("MCP HTTP Transport Error: empty response body")
	}

	if trimmed[0] == '[' {
		var msgs []jsonRPCMessage
		if err := json.Unmarshal(trimmed, &msgs); err != nil {
			return nil, fmt.Errorf("MCP HTTP Transport Error: parse JSON array: %w", err)
		}
		var result json.RawMessage
		var matched bool
		for _, m := range msgs {
			if m.ID != nil && *m.ID == requestID {
				if m.Error != nil {
					return nil, fmt.Errorf("RPC error %d: %s", m.Error.Code, m.Error.Message)
				}
				result = m.Result
				matched = true
				continue
			}
			if m.ID == nil && m.Method != "" {
				t.invokeNotifyHandler(m.Method, m.Params)
			}
		}
		if !matched {
			return nil, fmt.Errorf("MCP HTTP Transport Error: response missing id %d", requestID)
		}
		return result, nil
	}

	var msg jsonRPCMessage
	if err := json.Unmarshal(trimmed, &msg); err != nil {
		return nil, fmt.Errorf("MCP HTTP Transport Error: parse JSON: %w", err)
	}
	if msg.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", msg.Error.Code, msg.Error.Message)
	}
	return msg.Result, nil
}

// parseSSEResponse consumes the SSE body until the message that matches
// our request id arrives, dispatching everything else as notifications.
func (t *HTTPTransport) parseSSEResponse(body io.Reader, requestID int64) (json.RawMessage, error) {
	resultCh := make(chan json.RawMessage, 1)
	errCh := make(chan error, 1)

	go func() {
		err := scanSSE(body, func(eventType, data, _ string) {
			if eventType != "" && eventType != "message" {
				return
			}
			var m jsonRPCMessage
			if err := json.Unmarshal([]byte(data), &m); err != nil {
				return
			}
			if m.ID != nil && *m.ID == requestID {
				if m.Error != nil {
					select {
					case errCh <- fmt.Errorf("RPC error %d: %s", m.Error.Code, m.Error.Message):
					default:
					}
					return
				}
				select {
				case resultCh <- m.Result:
				default:
				}
				return
			}
			if m.ID == nil && m.Method != "" {
				t.invokeNotifyHandler(m.Method, m.Params)
			}
		})
		if err != nil {
			select {
			case errCh <- err:
			default:
			}
		} else {
			select {
			case errCh <- errors.New("MCP HTTP Transport Error: SSE stream closed before matching response"):
			default:
			}
		}
	}()

	select {
	case r := <-resultCh:
		return r, nil
	case err := <-errCh:
		return nil, err
	}
}

// OnNotification registers a handler for server notifications.
func (t *HTTPTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyMu.Lock()
	t.notifyHandler = handler
	t.notifyMu.Unlock()
}

func (t *HTTPTransport) invokeNotifyHandler(method string, params json.RawMessage) {
	t.notifyMu.Lock()
	h := t.notifyHandler
	t.notifyMu.Unlock()
	if h != nil {
		h(method, params)
	}
}

// runInboundSSE opens a long-lived GET to the server and dispatches any
// `message` events as notifications. Self-reconnects with exponential
// backoff and a `last-event-id` resume header (capped at 2 retries).
// Mirrors mcp-http-transport.ts:283-429.
func (t *HTTPTransport) runInboundSSE(ctx context.Context, resumeID string) {
	const maxRetries = 2
	const initialDelay = time.Second
	const maxDelay = 30 * time.Second
	const growth = 1.5

	attempts := 0
	lastID := resumeID

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		lastSeen, status := t.openInboundOnce(ctx, lastID)
		if status == inboundUnsupported || ctx.Err() != nil {
			return
		}
		if lastSeen != "" {
			lastID = lastSeen
		}

		if attempts >= maxRetries {
			return
		}
		delay := time.Duration(float64(initialDelay) * math.Pow(growth, float64(attempts)))
		if delay > maxDelay {
			delay = maxDelay
		}
		attempts++

		select {
		case <-ctx.Done():
			return
		case <-time.After(delay):
		}
	}
}

type inboundStatus int

const (
	inboundEnded       inboundStatus = iota // stream closed normally; reconnect
	inboundUnsupported                      // 405 — give up silently
	inboundError                            // network/parse error; reconnect
)

func (t *HTTPTransport) openInboundOnce(ctx context.Context, resumeID string) (string, inboundStatus) {
	extra := http.Header{"Accept": []string{"text/event-stream"}}
	if resumeID != "" {
		extra.Set(HeaderLastEventID, resumeID)
	}
	headers, err := t.commonHeaders(ctx, extra)
	if err != nil {
		return "", inboundError
	}

	req, err := http.NewRequestWithContext(ctx, "GET", t.url, nil)
	if err != nil {
		return "", inboundError
	}
	for k, vs := range headers {
		req.Header[k] = vs
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return "", inboundError
	}

	if resp.StatusCode == http.StatusUnauthorized && t.authProvider != nil {
		resourceMetaURL := ExtractResourceMetadataURL(resp)
		resp.Body.Close()
		res, err := Auth(ctx, t.authProvider, AuthOptions{
			ServerURL:           t.url,
			ResourceMetadataURL: resourceMetaURL,
		})
		if err != nil || res != AuthResultAuthorized {
			return "", inboundUnsupported
		}
		return t.openInboundOnce(ctx, resumeID)
	}

	if resp.StatusCode == http.StatusMethodNotAllowed {
		resp.Body.Close()
		return "", inboundUnsupported
	}

	if resp.StatusCode != http.StatusOK || resp.Body == nil {
		resp.Body.Close()
		return "", inboundError
	}

	if sid := resp.Header.Get(HeaderSessionID); sid != "" {
		t.sessionID.Store(sid)
	}

	defer resp.Body.Close()
	var lastID string
	_ = scanSSE(resp.Body, func(eventType, data, id string) {
		if id != "" {
			lastID = id
		}
		if eventType != "" && eventType != "message" {
			return
		}
		var m jsonRPCMessage
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			return
		}
		if m.ID == nil && m.Method != "" {
			t.invokeNotifyHandler(m.Method, m.Params)
		}
	})
	return lastID, inboundEnded
}

// resolveURL is exposed for tests that build expected URLs against the
// transport's base.
func (t *HTTPTransport) resolveURL(rel string) (string, error) {
	base, err := url.Parse(t.url)
	if err != nil {
		return "", err
	}
	r, err := url.Parse(rel)
	if err != nil {
		return "", err
	}
	return base.ResolveReference(r).String(), nil
}
