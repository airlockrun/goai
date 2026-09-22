package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/airlockrun/goai/tool"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestSessionStdio(t *testing.T) {
	if os.Getenv("GOAI_MCP_CHILD") == "1" {
		s := sdk.NewServer(&sdk.Implementation{Name: "child", Version: "1"}, nil)
		s.AddTool(&sdk.Tool{Name: "echo", InputSchema: json.RawMessage(`{"type":"object"}`)}, func(_ context.Context, r *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: string(r.Params.Arguments)}}}, nil
		})
		if err := s.Run(context.Background(), &sdk.StdioTransport{}); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	s, err := Connect(ctx, ServerConfig{Transport: "stdio", Command: os.Args[0], Args: []string{"-test.run=^TestSessionStdio$"}, Env: map[string]string{"GOAI_MCP_CHILD": "1"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	tools, err := s.ListTools(ctx)
	if err != nil || len(tools) != 1 {
		t.Fatalf("list %v %v", tools, err)
	}
	r, err := s.CallTool(ctx, "echo", []byte(`{"value":42}`))
	if err != nil || r.Content[0].(*sdk.TextContent).Text != `{"value":42}` {
		t.Fatalf("call %v %v", r, err)
	}
}

func TestSessionSSE(t *testing.T) {
	s := sdk.NewServer(&sdk.Implementation{Name: "sse", Version: "1"}, nil)
	s.AddTool(&sdk.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "ok"}}}, nil
	})
	h := httptest.NewServer(sdk.NewSSEHandler(func(*http.Request) *sdk.Server { return s }, nil))
	defer h.Close()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	client, err := Connect(ctx, ServerConfig{Transport: "sse", URL: h.URL, HTTPClient: h.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	r, err := client.CallTool(ctx, "lookup", []byte(`{}`))
	if err != nil || r.Content[0].(*sdk.TextContent).Text != "ok" {
		t.Fatalf("call %v %v", r, err)
	}
}

func TestSessionInputRequiredIsNotReplayed(t *testing.T) {
	var calls atomic.Int32
	h := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		result := map[string]any{"resultType": "complete", "supportedVersions": []string{"2026-07-28"}, "capabilities": map[string]any{"tools": map[string]any{}}}
		if request.Method == "tools/call" {
			calls.Add(1)
			result = map[string]any{"resultType": "input_required", "inputRequests": map[string]any{}, "requestState": "opaque"}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result})
	}))
	defer h.Close()
	s, err := Connect(t.Context(), ServerConfig{Transport: "http", URL: h.URL, HTTPClient: h.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	r, err := s.CallTool(t.Context(), "test", []byte(`{}`))
	if err != nil || !r.NeedsInput() || r.RequestState != "opaque" || calls.Load() != 1 {
		t.Fatalf("result %+v error %v calls %d", r, err, calls.Load())
	}
}

func TestSessionCloseCancelsCall(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	h := testServer(t, "2026-07-28", func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		close(started)
		<-release
		return &sdk.CallToolResult{}, nil
	})
	defer close(release)
	s, err := Connect(t.Context(), ServerConfig{Transport: "http", URL: h.URL, HTTPClient: h.Client()})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := s.CallTool(t.Context(), "lookup", []byte(`{}`)); done <- err }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("call did not start")
	}
	closed := make(chan error, 1)
	go func() { closed <- s.Close() }()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("call error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("call did not cancel")
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("close blocked")
	}
}

func TestClientListChangedReplacesSnapshot(t *testing.T) {
	server := sdk.NewServer(&sdk.Implementation{Name: "changes", Version: "1"}, nil)
	server.AddTool(&sdk.Tool{Name: "first", InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		return &sdk.CallToolResult{}, nil
	})
	h := httptest.NewServer(sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return server }, &sdk.StreamableHTTPOptions{Stateless: true, JSONResponse: true}))
	defer h.Close()
	c := NewClient()
	defer c.DisconnectAll()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	if err := c.Connect(ctx, ServerConfig{Name: "test", Transport: "http", URL: h.URL, HTTPClient: h.Client()}); err != nil {
		t.Fatal(err)
	}
	initial, err := c.GetTools(ctx)
	if err != nil || len(initial) != 1 {
		t.Fatalf("initial %v %v", initial, err)
	}
	server.RemoveTools("first")
	for {
		tools, err := c.GetTools(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if len(tools) == 0 {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("list removal not observed")
		case <-time.After(10 * time.Millisecond):
		}
	}
	if _, err := initial["test_first"].Execute(ctx, []byte(`{}`), tool.CallOptions{}); err == nil {
		t.Fatal("removed tool executed through stale closure")
	}
}

func TestDiscoveryBounds(t *testing.T) {
	for _, mode := range []string{"cycle", "pages", "bytes"} {
		t.Run(mode, func(t *testing.T) {
			count := 0
			err := pages(t.Context(), func(string) (string, any, error) {
				count++
				switch mode {
				case "cycle":
					return "same", nil, nil
				case "bytes":
					return "next", strings.Repeat("x", MaxHTTPResponseBytes), nil
				default:
					return strings.Repeat("p", count), nil, nil
				}
			})
			if err == nil || count > 128 {
				t.Fatalf("bounds: calls=%d error=%v", count, err)
			}
		})
	}
}

func TestSessionRejectsMalformedResources(t *testing.T) {
	h := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		result := map[string]any{"resultType": "complete", "supportedVersions": []string{"2026-07-28"}, "capabilities": map[string]any{"resources": map[string]any{}}}
		if request.Method == "resources/list" {
			result = map[string]any{"resultType": "complete", "resources": []any{nil}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result})
	}))
	defer h.Close()
	s, err := Connect(t.Context(), ServerConfig{Transport: "http", URL: h.URL, HTTPClient: h.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.ListResources(t.Context()); err == nil {
		t.Fatal("malformed resources accepted")
	}
}
