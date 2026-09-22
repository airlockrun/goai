package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/tool"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestClientConnectUsesConfiguredHTTPClient(t *testing.T) {
	want := errors.New("configured client used")
	c := NewClient()
	err := c.Connect(t.Context(), ServerConfig{Name: "test", Transport: "http", URL: "https://example.test/mcp", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, want })}})
	if !errors.Is(err, want) {
		t.Fatalf("Connect: %v", err)
	}
}

func testServer(t *testing.T, version string, handler sdk.ToolHandler) *httptest.Server {
	t.Helper()
	s := sdk.NewServer(&sdk.Implementation{Name: "test", Version: "1"}, &sdk.ServerOptions{Instructions: "Test instructions", SupportedProtocolVersions: []string{version}})
	s.AddTool(&sdk.Tool{Name: "lookup", Description: "Lookup", InputSchema: json.RawMessage(`{"type":"object","properties":{"id":{"type":"integer"}}}`), OutputSchema: json.RawMessage(`{"type":"object"}`)}, handler)
	s.AddResource(&sdk.Resource{URI: "file:///test.txt", Name: "test", MIMEType: "text/plain"}, func(context.Context, *sdk.ReadResourceRequest) (*sdk.ReadResourceResult, error) {
		return &sdk.ReadResourceResult{Contents: []*sdk.ResourceContents{{URI: "file:///test.txt", MIMEType: "text/plain", Text: "resource content"}}}, nil
	})
	h := httptest.NewServer(sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return s }, &sdk.StreamableHTTPOptions{Stateless: true, JSONResponse: true}))
	t.Cleanup(h.Close)
	return h
}

func TestClientToolsAndResources(t *testing.T) {
	for _, version := range []string{"2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"} {
		t.Run(version, func(t *testing.T) {
			h := testServer(t, version, func(_ context.Context, r *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				if string(r.Params.Arguments) != `{"id":9007199254740993}` {
					t.Errorf("arguments changed: %s", r.Params.Arguments)
				}
				return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "found"}, &sdk.ImageContent{Data: []byte("image"), MIMEType: "image/png"}, &sdk.AudioContent{Data: []byte("audio"), MIMEType: "audio/wav"}}}, nil
			})
			c := NewClient()
			defer c.DisconnectAll()
			if err := c.Connect(t.Context(), ServerConfig{Name: "test", Transport: "http", URL: h.URL, HTTPClient: h.Client()}); err != nil {
				t.Fatal(err)
			}
			tools, err := c.GetTools(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			definition, ok := tools["test_lookup"]
			if !ok || len(definition.OutputSchema) == 0 {
				t.Fatalf("definition: %+v", definition)
			}
			result, err := definition.Execute(t.Context(), json.RawMessage(`{"id":9007199254740993}`), tool.CallOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output != "found" || len(result.Attachments) != 2 {
				t.Fatalf("projection: %+v", result)
			}
			if c.GetServerInstructions("test") != "Test instructions" {
				t.Fatal("instructions lost")
			}
			resources, err := c.GetResources(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if len(resources) != 1 {
				t.Fatal("resources lost")
			}
			r, err := c.ReadResource(t.Context(), "file:///test.txt")
			if err != nil || r.Text != "resource content" {
				t.Fatalf("read: %+v %v", r, err)
			}
			if err := c.Disconnect("test"); err != nil {
				t.Fatal(err)
			}
			tools, err = c.GetTools(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if len(tools) != 0 {
				t.Fatal("tools retained after close")
			}
		})
	}
}

func TestProjectResult(t *testing.T) {
	for _, tc := range []struct {
		name, raw, want string
		isError         bool
	}{
		{"structured", `{"structuredContent":{"value":42}}`, `{"value":42}`, false},
		{"error", `{"isError":true,"content":[{"type":"text","text":"refused"}]}`, "refused", true},
		{"embedded", `{"content":[{"type":"resource","resource":{"text":"embedded"}}]}`, "embedded", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := projectResult([]byte(tc.raw))
			if (err != nil) != tc.isError || r.Output != tc.want {
				t.Fatalf("result %+v error %v", r, err)
			}
		})
	}
}
