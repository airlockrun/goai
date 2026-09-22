package mcp

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// sessionHTTPClient preserves the supplied network and redirect policies.
func sessionHTTPClient(config ServerConfig) (*http.Client, error) {
	endpoint, err := url.Parse(config.URL)
	if err != nil {
		return nil, err
	}
	if endpoint.Host == "" || (endpoint.Scheme != "https" && endpoint.Scheme != "http") {
		return nil, errors.New("MCP endpoint must be an absolute HTTP URL")
	}
	client := *config.HTTPClient
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	headers := make(map[string]string, len(config.Headers))
	for key, value := range config.Headers {
		headers[key] = value
	}
	client.Transport = &sessionTransport{base: base, endpoint: endpoint, headers: headers, provider: config.AuthProvider, authClient: config.HTTPClient, sse: config.Transport == "sse"}
	return &client, nil
}

type sessionTransport struct {
	base       http.RoundTripper
	endpoint   *url.URL
	headers    map[string]string
	provider   OAuthClientProvider
	authClient *http.Client
	sse        bool
	authMu     sync.Mutex
}

func (t *sessionTransport) RoundTrip(original *http.Request) (*http.Response, error) {
	for attempt := 0; attempt < 2; attempt++ {
		req := original.Clone(original.Context())
		if !strings.Contains(req.UserAgent(), UserAgentSuffix) {
			req.Header.Set("User-Agent", strings.TrimSpace(req.UserAgent()+" "+UserAgentSuffix))
		}
		sameOrigin := strings.EqualFold(req.URL.Scheme, t.endpoint.Scheme) && strings.EqualFold(req.URL.Host, t.endpoint.Host)
		if t.sse && !sameOrigin {
			return nil, errors.New("MCP SSE endpoint must have the configured origin")
		}
		for key, value := range t.headers {
			canonical := http.CanonicalHeaderKey(key)
			if strings.HasPrefix(canonical, "Mcp-") || canonical == "Content-Type" || canonical == "Accept" || canonical == "Last-Event-Id" {
				continue
			}
			if sameOrigin {
				req.Header.Set(key, value)
			} else {
				req.Header.Del(key)
			}
		}
		if t.provider != nil {
			req.Header.Del("Authorization")
			if sameOrigin {
				tokens, err := t.provider.Tokens(req.Context())
				if err != nil {
					return nil, err
				}
				if tokens != nil && tokens.AccessToken != "" {
					req.Header.Set("Authorization", "Bearer "+tokens.AccessToken)
				}
			}
		}
		if attempt > 0 && original.Body != nil {
			if original.GetBody == nil {
				return nil, errors.New("MCP authorization retry requires replayable request body")
			}
			body, err := original.GetBody()
			if err != nil {
				return nil, err
			}
			req.Body = body
		}
		resp, err := t.base.RoundTrip(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusUnauthorized && t.provider != nil && sameOrigin && attempt == 0 {
			metadata := ExtractResourceMetadataURL(resp)
			resp.Body.Close()
			t.authMu.Lock()
			// Reuse tokens refreshed by another request while this request waited.
			tokens, err := t.provider.Tokens(req.Context())
			if err == nil && (tokens == nil || tokens.AccessToken == "" || "Bearer "+tokens.AccessToken == req.Header.Get("Authorization")) {
				var result AuthResult
				result, err = Auth(req.Context(), t.provider, AuthOptions{ServerURL: t.endpoint.String(), ResourceMetadataURL: metadata, HTTPClient: t.authClient})
				if err == nil && result != AuthResultAuthorized {
					err = &UnauthorizedError{Message: "authorization redirect required"}
				}
			}
			t.authMu.Unlock()
			if err != nil {
				return nil, err
			}
			continue
		}
		limit := int64(MaxHTTPResponseBytes)
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			limit = 64 << 10
		}
		resp.Body = &boundedBody{ReadCloser: resp.Body, remaining: limit}
		if resp.StatusCode >= 400 && req.Method == http.MethodPost && !(resp.StatusCode == http.StatusNotFound && req.Header.Get("Mcp-Session-Id") != "") {
			body, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				return nil, readErr
			}
			var envelope struct {
				JSONRPC string          `json:"jsonrpc"`
				Error   json.RawMessage `json:"error"`
			}
			if json.Unmarshal(body, &envelope) != nil || envelope.JSONRPC != "2.0" || len(envelope.Error) == 0 {
				return nil, &MCPClientError{Message: fmt.Sprintf("MCP HTTP %d %s", resp.StatusCode, http.StatusText(resp.StatusCode)), StatusCode: resp.StatusCode, URL: req.URL.String(), ResponseBody: string(body)}
			}
			resp.Body = io.NopCloser(bytes.NewReader(body))
		}
		return resp, nil
	}
	return nil, errors.New("MCP authorization failed")
}

type boundedBody struct {
	io.ReadCloser
	remaining int64
}

func (b *boundedBody) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if b.remaining == 0 {
		var probe [1]byte
		n, err := b.ReadCloser.Read(probe[:])
		if n > 0 {
			return 0, errors.New("MCP HTTP response exceeds byte limit")
		}
		return 0, err
	}
	if int64(len(p)) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.ReadCloser.Read(p)
	b.remaining -= int64(n)
	return n, err
}
