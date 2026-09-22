package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type synchronizedProvider struct {
	stubProvider
	mu sync.Mutex
}

func (p *synchronizedProvider) Tokens(context.Context) (*OAuthTokens, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	copy := *p.tokens
	return &copy, nil
}
func (p *synchronizedProvider) SaveTokens(_ context.Context, tokens *OAuthTokens) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tokens = tokens
	return nil
}

func TestHTTPAuthorizationRefreshIsShared(t *testing.T) {
	var refreshes atomic.Int32
	var stale atomic.Int32
	allStale := make(chan struct{})
	h := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/.well-known/oauth-protected-resource":
			_ = json.NewEncoder(w).Encode(map[string]any{"resource": "http://" + r.Host, "authorization_servers": []string{"http://" + r.Host}})
		case "/.well-known/oauth-authorization-server":
			_ = json.NewEncoder(w).Encode(AuthorizationServerMetadata{Issuer: "http://" + r.Host, AuthorizationEndpoint: "http://" + r.Host + "/auth", TokenEndpoint: "http://" + r.Host + "/token", ResponseTypesSupported: []string{"code"}, CodeChallengeMethodsSupported: []string{"S256"}})
		case "/token":
			refreshes.Add(1)
			_ = json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "fresh", TokenType: "Bearer", RefreshToken: "next"})
		default:
			if r.Header.Get("Authorization") == "Bearer stale" {
				if stale.Add(1) == 2 {
					close(allStale)
				}
				<-allStale
				w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="http://`+r.Host+`/.well-known/oauth-protected-resource"`)
				w.WriteHeader(401)
				return
			}
			if r.Header.Get("Authorization") != "Bearer fresh" {
				t.Error("missing refreshed credential")
			}
			w.WriteHeader(204)
		}
	}))
	defer h.Close()
	provider := &synchronizedProvider{stubProvider: stubProvider{tokens: &OAuthTokens{AccessToken: "stale", RefreshToken: "refresh"}, clientInfo: &OAuthClientInformation{ClientID: "client"}, redirectURL: "http://localhost/callback"}}
	client, err := sessionHTTPClient(ServerConfig{Transport: "http", URL: h.URL, HTTPClient: h.Client(), AuthProvider: provider})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		wg.Go(func() {
			req, err := http.NewRequestWithContext(t.Context(), method, h.URL, strings.NewReader("{}"))
			if err != nil {
				t.Error(err)
				return
			}
			resp, err := client.Do(req)
			if err != nil {
				t.Error(err)
				return
			}
			resp.Body.Close()
		})
	}
	wg.Wait()
	if refreshes.Load() != 1 {
		t.Fatalf("refresh calls=%d", refreshes.Load())
	}
}

func TestSessionHTTPPolicies(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" || r.Header.Get("X-API-Key") != "" {
			t.Error("credentials leaked")
		}
		w.WriteHeader(204)
	}))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-API-Key") != "secret" {
			t.Error("missing source credential")
		}
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	client, err := sessionHTTPClient(ServerConfig{URL: source.URL, Transport: "http", HTTPClient: source.Client(), Headers: map[string]string{"Authorization": "Bearer secret", "X-API-Key": "secret", "Mcp-Method": "spoof"}})
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, source.URL, strings.NewReader("{}"))
	req.Header.Set("Mcp-Method", "server/discover")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	blocked := source.Client()
	blocked.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	client, err = sessionHTTPClient(ServerConfig{URL: source.URL, Transport: "http", HTTPClient: blocked, Headers: map[string]string{"X-API-Key": "secret"}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err = client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 307 {
		t.Fatal("redirect policy lost")
	}
}

func TestBoundedBody(t *testing.T) {
	for _, tc := range []struct {
		body string
		fail bool
	}{{"1234", false}, {"12345", true}} {
		t.Run(tc.body, func(t *testing.T) {
			body := &boundedBody{ReadCloser: io.NopCloser(strings.NewReader(tc.body)), remaining: 4}
			got, err := io.ReadAll(body)
			if (err != nil) != tc.fail || string(got) != "1234" {
				t.Fatalf("read %q %v", got, err)
			}
		})
	}
}

func TestHTTPUnauthorizedClassification(t *testing.T) {
	h := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "unauthorized", 401) }))
	defer h.Close()
	_, err := Connect(t.Context(), ServerConfig{Transport: "http", URL: h.URL, HTTPClient: h.Client()})
	var typed *MCPClientError
	if !errors.As(err, &typed) || typed.StatusCode != 401 {
		t.Fatalf("error: %v", err)
	}
}
