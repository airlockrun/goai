package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// startSSEServer spins up a minimal MCP SSE server. Tests configure
// behavior via the returned hooks before issuing requests.
func startSSEServer(t *testing.T, opts sseServerOpts) *sseRunningServer {
	t.Helper()
	rs := &sseRunningServer{opts: opts, sseSink: make(chan string, 16), done: make(chan struct{})}
	mux := http.NewServeMux()

	mux.HandleFunc("/sse", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			rs.recordGET(r)
			if rs.opts.GetStatus != 0 {
				w.WriteHeader(rs.opts.GetStatus)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			fl := w.(http.Flusher)
			fl.Flush()

			if !rs.opts.SkipEndpoint {
				if rs.opts.EndpointDelay > 0 {
					time.Sleep(rs.opts.EndpointDelay)
				}
				ep := rs.opts.EndpointPath
				if ep == "" {
					ep = "/messages/abc"
				}
				if rs.opts.AbsoluteEndpoint != "" {
					ep = rs.opts.AbsoluteEndpoint
				}
				fmt.Fprintf(w, "event: endpoint\ndata: %s\n\n", ep)
				fl.Flush()
			}
			// Then drain sseSink (server pushes messages to clients here)
			for {
				select {
				case <-r.Context().Done():
					return
				case <-rs.done:
					return
				case msg := <-rs.sseSink:
					fmt.Fprint(w, msg)
					fl.Flush()
				}
			}
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/messages/abc", rs.handlePost)

	srv := httptest.NewServer(mux)
	rs.Server = srv
	t.Cleanup(func() {
		close(rs.done)
		srv.Close()
	})
	return rs
}

type sseServerOpts struct {
	EndpointPath     string
	AbsoluteEndpoint string
	EndpointDelay    time.Duration
	SkipEndpoint     bool
	GetStatus        int
}

type sseRunningServer struct {
	*httptest.Server
	opts    sseServerOpts
	sseSink chan string
	done    chan struct{}

	mu           sync.Mutex
	getRequests  []*http.Request
	postRequests []*http.Request
	postBodies   [][]byte
}

func (rs *sseRunningServer) recordGET(r *http.Request) {
	rs.mu.Lock()
	rs.getRequests = append(rs.getRequests, r.Clone(r.Context()))
	rs.mu.Unlock()
}

func (rs *sseRunningServer) handlePost(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	rs.mu.Lock()
	rs.postRequests = append(rs.postRequests, r.Clone(r.Context()))
	rs.postBodies = append(rs.postBodies, body)
	rs.mu.Unlock()

	// Push the response back over the SSE channel.
	var req struct {
		ID int64 `json:"id"`
	}
	_ = json.Unmarshal(body, &req)
	rs.sseSink <- fmt.Sprintf("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":\"ok\"}\n\n", req.ID)
	w.WriteHeader(http.StatusAccepted)
}

func (rs *sseRunningServer) recorded() ([]*http.Request, []*http.Request, [][]byte) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	gets := append([]*http.Request(nil), rs.getRequests...)
	posts := append([]*http.Request(nil), rs.postRequests...)
	bodies := append([][]byte(nil), rs.postBodies...)
	return gets, posts, bodies
}

func TestSSETransport_ProtocolVersionHeader(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "ping", nil); err != nil {
		t.Fatal(err)
	}
	gets, posts, _ := rs.recorded()
	for _, r := range gets {
		if r.Header.Get(HeaderProtocolVersion) != LatestProtocolVersion {
			t.Errorf("GET missing protocol version (got %q)", r.Header.Get(HeaderProtocolVersion))
		}
	}
	for _, r := range posts {
		if r.Header.Get(HeaderProtocolVersion) != LatestProtocolVersion {
			t.Errorf("POST missing protocol version (got %q)", r.Header.Get(HeaderProtocolVersion))
		}
	}
}

func TestSSETransport_EndpointEventHandshake(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{EndpointPath: "/messages/abc"})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "ping", nil); err != nil {
		t.Fatal(err)
	}
	_, posts, _ := rs.recorded()
	if len(posts) == 0 {
		t.Fatal("no POSTs recorded")
	}
	if !strings.HasSuffix(posts[0].URL.Path, "/messages/abc") {
		t.Errorf("POST went to %s, expected /messages/abc", posts[0].URL.Path)
	}
}

func TestSSETransport_EndpointOriginMismatch(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{AbsoluteEndpoint: "https://other.example.com/messages"})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	err := tr.Connect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "endpoint origin") {
		t.Errorf("expected origin mismatch error, got %v", err)
	}
}

func TestSSETransport_EndpointEventRelativePath(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{EndpointPath: "/messages/abc"})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Send(context.Background(), "ping", nil); err != nil {
		t.Fatal(err)
	}
	_, posts, _ := rs.recorded()
	if len(posts) == 0 {
		t.Fatal("no POSTs")
	}
	want := rs.URL + "/messages/abc"
	got := "http://" + posts[0].Host + posts[0].URL.Path
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestSSETransport_ConnectBlocksUntilEndpoint(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{EndpointDelay: 80 * time.Millisecond})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	start := time.Now()
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if elapsed := time.Since(start); elapsed < 60*time.Millisecond {
		t.Errorf("Connect returned in %s, expected to wait for endpoint event", elapsed)
	}
}

func TestSSETransport_ConnectFailsOnStreamCloseBeforeEndpoint(t *testing.T) {
	rs := startSSEServer(t, sseServerOpts{SkipEndpoint: true})
	tr := NewSSETransport(rs.URL+"/sse", nil, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	err := tr.Connect(ctx)
	if err == nil {
		t.Fatal("expected error when stream closes before endpoint event")
	}
}

func TestSSETransport_AuthOn401Retry(t *testing.T) {
	// Spin up an integrated mux: SSE server + OAuth metadata + token endpoint.
	mux := http.NewServeMux()
	var firstGet atomic.Bool
	firstGet.Store(true)

	sseSink := make(chan string, 16)
	mux.HandleFunc("/sse", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if firstGet.CompareAndSwap(true, false) {
			w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="http://`+r.Host+`/.well-known/oauth-protected-resource"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Authorization") != "Bearer fresh" {
			http.Error(w, "missing bearer", 401)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		fmt.Fprintf(w, "event: endpoint\ndata: /messages/abc\n\n")
		fl.Flush()
		for {
			select {
			case <-r.Context().Done():
				return
			case msg := <-sseSink:
				fmt.Fprint(w, msg)
				fl.Flush()
			}
		}
	})
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resource":              "http://" + r.Host + "/sse",
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
	tr := NewSSETransport(srv.URL+"/sse", nil, prov)
	if err := tr.Connect(context.Background()); err != nil {
		t.Fatalf("connect after auth retry should succeed: %v", err)
	}
	defer tr.Close()
	if prov.tokens.AccessToken != "fresh" {
		t.Errorf("expected refreshed token, got %s", prov.tokens.AccessToken)
	}
}
