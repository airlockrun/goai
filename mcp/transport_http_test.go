package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// httpTestServer wraps httptest.Server with a hook for asserting per-
// request shape and a per-method dispatch table.
type httpTestServer struct {
	*httptest.Server
	mu        sync.Mutex
	requests  []*http.Request
	bodies    [][]byte
	postFn    func(w http.ResponseWriter, r *http.Request)
	getFn     func(w http.ResponseWriter, r *http.Request)
	deleteFn  func(w http.ResponseWriter, r *http.Request)
	sessionID string
}

func newHTTPTestServer(t *testing.T) *httpTestServer {
	hts := &httpTestServer{}
	hts.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(strings.NewReader(string(body)))
		hts.mu.Lock()
		hts.requests = append(hts.requests, r.Clone(r.Context()))
		hts.bodies = append(hts.bodies, body)
		hts.mu.Unlock()
		switch r.Method {
		case "POST":
			if hts.postFn != nil {
				hts.postFn(w, r)
				return
			}
			http.Error(w, "no postFn", 500)
		case "GET":
			if hts.getFn != nil {
				hts.getFn(w, r)
				return
			}
			w.WriteHeader(http.StatusMethodNotAllowed)
		case "DELETE":
			if hts.deleteFn != nil {
				hts.deleteFn(w, r)
				return
			}
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "bad method", 400)
		}
	}))
	return hts
}

func (h *httpTestServer) recorded() ([]*http.Request, [][]byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	reqs := append([]*http.Request(nil), h.requests...)
	bodies := append([][]byte(nil), h.bodies...)
	return reqs, bodies
}

// reply writes a single JSON-RPC result message keyed to whatever id the
// caller sent.
func reply(w http.ResponseWriter, body []byte, result any) {
	var req struct {
		ID int64 `json:"id"`
	}
	_ = json.Unmarshal(body, &req)
	w.Header().Set("Content-Type", "application/json")
	resBytes, _ := json.Marshal(result)
	json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      req.ID,
		"result":  json.RawMessage(resBytes),
	})
}

func TestHTTPTransport_ProtocolVersionHeader(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reply(w, body, map[string]string{"ok": "yes"})
	}

	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "tools/list", nil); err != nil {
		t.Fatal(err)
	}

	reqs, _ := hts.recorded()
	for _, r := range reqs {
		if r.Header.Get(HeaderProtocolVersion) != LatestProtocolVersion {
			t.Errorf("%s %s missing protocol version (got %q)", r.Method, r.URL.Path, r.Header.Get(HeaderProtocolVersion))
		}
	}
}

func TestHTTPTransport_UserAgentSuffix(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "ping", nil); err != nil {
		t.Fatal(err)
	}
	reqs, _ := hts.recorded()
	if len(reqs) == 0 {
		t.Fatal("no requests recorded")
	}
	for _, r := range reqs {
		ua := r.Header.Get("User-Agent")
		if !strings.Contains(ua, UserAgentSuffix) {
			t.Errorf("UA %q missing %q", ua, UserAgentSuffix)
		}
	}
}

func TestHTTPTransport_SessionIDRoundtrip(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	var seen atomic.Int32
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		count := seen.Add(1)
		if count == 1 {
			// First request: server hands out a session id.
			w.Header().Set(HeaderSessionID, "sess-abc")
		} else {
			// Subsequent: client should echo back what the server set.
			if got := r.Header.Get(HeaderSessionID); got != "sess-abc" {
				t.Errorf("subsequent request session id = %q, want sess-abc", got)
			}
		}
		reply(w, body, "ok")
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "initialize", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Send(context.Background(), "tools/list", nil); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPTransport_DeleteOnClose(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(HeaderSessionID, "S")
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	}
	deletes := make(chan *http.Request, 1)
	hts.deleteFn = func(w http.ResponseWriter, r *http.Request) {
		deletes <- r
		w.WriteHeader(http.StatusOK)
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Send(context.Background(), "initialize", nil); err != nil {
		t.Fatal(err)
	}
	tr.Close()
	select {
	case r := <-deletes:
		if r.Header.Get(HeaderSessionID) != "S" {
			t.Errorf("DELETE missing session id, got %q", r.Header.Get(HeaderSessionID))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("DELETE not received within timeout")
	}
}

func TestHTTPTransport_ContentTypeSSEResponse(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID int64 `json:"id"`
		}
		_ = json.Unmarshal(body, &req)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		fmt.Fprintf(w, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":{\"k\":\"v\"}}\n\n", req.ID)
		fl.Flush()
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	got, err := tr.Send(context.Background(), "any", nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"k":"v"}` {
		t.Errorf("got %s", string(got))
	}
}

func TestHTTPTransport_ContentTypeJSONArray(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID int64 `json:"id"`
		}
		_ = json.Unmarshal(body, &req)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[{"jsonrpc":"2.0","id":%d,"result":{"k":1}},{"jsonrpc":"2.0","method":"notifications/ping"}]`, req.ID)
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	got, err := tr.Send(context.Background(), "x", nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"k":1}` {
		t.Errorf("got %s", string(got))
	}
}

func TestHTTPTransport_404HelpfulMessage(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) }
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	_, err := tr.Send(context.Background(), "tools/list", nil)
	if err == nil || !strings.Contains(err.Error(), "sse") {
		t.Errorf("expected sse hint, got %v", err)
	}
}

func TestHTTPTransport_MCPClientErrorHTTPFields(t *testing.T) {
	tests := []struct {
		name       string
		postFn     func(w http.ResponseWriter, r *http.Request)
		wantStatus int
		wantBody   string
		wantMsg    string
	}{
		{
			name: "500 carries status, url, body",
			postFn: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte("Internal Server Error"))
			},
			wantStatus: 500,
			wantBody:   "Internal Server Error",
			wantMsg:    "POSTing to endpoint",
		},
		{
			name: "404 carries status, url, body and sse hint",
			postFn: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte("Not Found"))
			},
			wantStatus: 404,
			wantBody:   "Not Found",
			wantMsg:    "does not support HTTP transport",
		},
		{
			name: "unexpected content type carries status and url",
			postFn: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/plain")
				w.WriteHeader(http.StatusOK)
				w.Write([]byte("nope"))
			},
			wantStatus: 200,
			wantBody:   "",
			wantMsg:    "unexpected content type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hts := newHTTPTestServer(t)
			defer hts.Close()
			hts.postFn = tt.postFn
			tr := NewHTTPTransport(hts.URL, nil, nil)
			if err := tr.Connect(context.Background()); err != nil {
				t.Fatal(err)
			}
			defer tr.Close()

			_, err := tr.Send(context.Background(), "tools/list", nil)
			var mcpErr *MCPClientError
			if !errors.As(err, &mcpErr) {
				t.Fatalf("expected *MCPClientError, got %T: %v", err, err)
			}
			if mcpErr.StatusCode != tt.wantStatus {
				t.Errorf("StatusCode = %d, want %d", mcpErr.StatusCode, tt.wantStatus)
			}
			if mcpErr.URL != hts.URL {
				t.Errorf("URL = %q, want %q", mcpErr.URL, hts.URL)
			}
			if mcpErr.ResponseBody != tt.wantBody {
				t.Errorf("ResponseBody = %q, want %q", mcpErr.ResponseBody, tt.wantBody)
			}
			if !strings.Contains(mcpErr.Message, tt.wantMsg) {
				t.Errorf("Message = %q, want substring %q", mcpErr.Message, tt.wantMsg)
			}
		})
	}
}

// fakeAuthProvider is a minimal token-bearing OAuthClientProvider that
// flips its access token on every Refresh, so we can verify auth-retry
// actually re-issued the request with new credentials.
type fakeAuthProvider struct {
	mu     sync.Mutex
	tokens *OAuthTokens
}

func (f *fakeAuthProvider) Tokens(ctx context.Context) (*OAuthTokens, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.tokens == nil {
		return nil, nil
	}
	t := *f.tokens
	return &t, nil
}
func (f *fakeAuthProvider) SaveTokens(ctx context.Context, t *OAuthTokens) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tokens = t
	return nil
}
func (f *fakeAuthProvider) RedirectToAuthorization(ctx context.Context, u *url.URL) error {
	return nil
}
func (f *fakeAuthProvider) SaveCodeVerifier(ctx context.Context, v string) error { return nil }
func (f *fakeAuthProvider) CodeVerifier(ctx context.Context) (string, error)     { return "v", nil }
func (f *fakeAuthProvider) RedirectURL() string                                  { return "https://app/cb" }
func (f *fakeAuthProvider) ClientMetadata() OAuthClientMetadata {
	return OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}}
}
func (f *fakeAuthProvider) ClientInformation(ctx context.Context) (*OAuthClientInformation, error) {
	return &OAuthClientInformation{ClientID: "cid"}, nil
}

func TestHTTPTransport_AuthOn401Retry(t *testing.T) {
	// Spin up a single mux that plays both the MCP server (returns 401
	// then 200) and an OAuth metadata + token endpoint. The fakeAuthProvider
	// already has a refresh token, so Auth() runs the refresh flow without
	// triggering a redirect.
	var attempt atomic.Int32

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		n := attempt.Add(1)
		if n == 1 {
			w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="http://`+r.Host+`/.well-known/oauth-protected-resource"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		// After auth retry, must carry new bearer.
		if r.Header.Get("Authorization") != "Bearer fresh" {
			http.Error(w, "missing bearer", 401)
			return
		}
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	})
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resource":              "http://" + r.Host + "/mcp",
			"authorization_servers": []string{"http://" + r.Host},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthorizationServerMetadata{
			Issuer:                        "http://" + r.Host,
			AuthorizationEndpoint:         "http://" + r.Host + "/auth",
			TokenEndpoint:                 "http://" + r.Host + "/token",
			ResponseTypesSupported:        []string{"code"},
			CodeChallengeMethodsSupported: []string{"S256"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "fresh", TokenType: "Bearer", RefreshToken: "rt2"})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	prov := &fakeAuthProvider{tokens: &OAuthTokens{AccessToken: "stale", TokenType: "Bearer", RefreshToken: "rt"}}

	tr := NewHTTPTransport(srv.URL+"/mcp", nil, prov)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()

	if _, err := tr.Send(context.Background(), "tools/list", nil); err != nil {
		t.Fatalf("expected retry to succeed: %v", err)
	}
	if attempt.Load() < 2 {
		t.Errorf("expected at least 2 POST attempts, got %d", attempt.Load())
	}
	if prov.tokens.AccessToken != "fresh" {
		t.Errorf("expected token refresh, got %s", prov.tokens.AccessToken)
	}
}

// TestHTTPTransport_AuthDedup drives the POST and inbound-GET 401 paths
// concurrently and asserts authorizeOnce collapses them into a single token
// refresh, mirroring ai-sdk's deduplicate-auth-refresh fix.
func TestHTTPTransport_AuthDedup(t *testing.T) {
	var tokenCalls atomic.Int32
	// release gates the token endpoint so both 401 paths are in-flight
	// before either refresh completes.
	release := make(chan struct{})

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer stale" {
			w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="http://`+r.Host+`/.well-known/oauth-protected-resource"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.Method == "GET" {
			// Hold the authorized inbound GET open until the test ends.
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			return
		}
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	})
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resource":              "http://" + r.Host + "/mcp",
			"authorization_servers": []string{"http://" + r.Host},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthorizationServerMetadata{
			Issuer:                        "http://" + r.Host,
			AuthorizationEndpoint:         "http://" + r.Host + "/auth",
			TokenEndpoint:                 "http://" + r.Host + "/token",
			ResponseTypesSupported:        []string{"code"},
			CodeChallengeMethodsSupported: []string{"S256"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		tokenCalls.Add(1)
		<-release
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "fresh", TokenType: "Bearer", RefreshToken: "rt2"})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	prov := &fakeAuthProvider{tokens: &OAuthTokens{AccessToken: "stale", TokenType: "Bearer", RefreshToken: "rt"}}

	tr := NewHTTPTransport(srv.URL+"/mcp", nil, prov)
	// Connect kicks off the inbound GET, which hits 401 and starts a refresh.
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()

	// Fire a POST concurrently; it also hits 401 and should join the
	// in-flight refresh rather than start its own.
	sendErr := make(chan error, 1)
	go func() {
		_, err := tr.Send(context.Background(), "tools/list", nil)
		sendErr <- err
	}()

	// Wait until at least one refresh is parked at the token endpoint, then
	// give the second 401 path time to attempt joining before releasing.
	deadline := time.After(2 * time.Second)
	for tokenCalls.Load() == 0 {
		select {
		case <-deadline:
			t.Fatal("token endpoint never called")
		case <-time.After(time.Millisecond):
		}
	}
	time.Sleep(50 * time.Millisecond)
	close(release)

	if err := <-sendErr; err != nil {
		t.Fatalf("POST after dedup auth failed: %v", err)
	}
	if got := tokenCalls.Load(); got != 1 {
		t.Errorf("token endpoint called %d times, want 1 (dedup)", got)
	}
}

func TestHTTPTransport_InboundSSE405Silent(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	}
	// hts.getFn unset → server replies 405. Transport should swallow.
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	// Give the inbound goroutine a moment to attempt the GET.
	time.Sleep(50 * time.Millisecond)
	// POST should still work.
	if _, err := tr.Send(context.Background(), "ping", nil); err != nil {
		t.Fatalf("transport should remain usable after 405: %v", err)
	}
}

func TestHTTPTransport_InboundSSENotifications(t *testing.T) {
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reply(w, body, "ok")
	}
	hts.getFn = func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		fmt.Fprint(w, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n")
		fl.Flush()
		// Hold the connection briefly so the client has time to read.
		time.Sleep(100 * time.Millisecond)
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	notifyCh := make(chan string, 1)
	tr.OnNotification(func(method string, _ json.RawMessage) {
		select {
		case notifyCh <- method:
		default:
		}
	})
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	select {
	case m := <-notifyCh:
		if m != "notifications/tools/list_changed" {
			t.Errorf("got %s", m)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("notification not received")
	}
}

func TestHTTPTransport_NotificationsSendNoWait(t *testing.T) {
	// Notifications must not deadlock waiting for a response — server
	// returns 202 with no body.
	hts := newHTTPTestServer(t)
	defer hts.Close()
	hts.postFn = func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}
	tr := NewHTTPTransport(hts.URL, nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	res, err := tr.Send(ctx, "notifications/initialized", nil)
	if err != nil {
		t.Fatal(err)
	}
	if res != nil {
		t.Errorf("expected nil result, got %s", string(res))
	}
}
