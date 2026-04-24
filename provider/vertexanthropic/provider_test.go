package vertexanthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/google-vertex/src/anthropic/
// google-vertex-anthropic-provider.test.ts.

func TestProvider_ID(t *testing.T) {
	p := New(Options{Project: "test-project", Location: "us-east5"})
	if got := p.ID(); got != "vertex.anthropic" {
		t.Errorf("ID() = %q, want %q", got, "vertex.anthropic")
	}
}

func TestProvider_Models(t *testing.T) {
	p := New(Options{Project: "test-project"})
	models := p.Models()
	if len(models) != len(VertexAnthropicChatModels) {
		t.Fatalf("Models() length = %d, want %d", len(models), len(VertexAnthropicChatModels))
	}
	// Spot-check a few entries that should be present.
	wants := []string{
		"claude-3-5-sonnet-v2@20241022",
		"claude-sonnet-4-5@20250929",
		"claude-opus-4-1@20250805",
	}
	for _, w := range wants {
		found := false
		for _, m := range models {
			if m == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Models() missing %q", w)
		}
	}
}

func TestProvider_BaseURL(t *testing.T) {
	tests := []struct {
		name    string
		opts    Options
		want    string
		wantErr bool
	}{
		{
			name: "region-prefixed for non-global location",
			opts: Options{Project: "test-project", Location: "us-east5"},
			want: "https://us-east5-aiplatform.googleapis.com/v1/projects/test-project/locations/us-east5/publishers/anthropic/models",
		},
		{
			name: "global location omits region prefix",
			opts: Options{Project: "test-project", Location: "global"},
			want: "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/anthropic/models",
		},
		{
			name: "default location is us-central1",
			opts: Options{Project: "test-project"},
			want: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/anthropic/models",
		},
		{
			name: "BaseURL override bypasses construction",
			opts: Options{Project: "test-project", Location: "us-east5", BaseURL: "https://custom.example.com/models"},
			want: "https://custom.example.com/models",
		},
		{
			name: "BaseURL override trims trailing slash",
			opts: Options{Project: "test-project", BaseURL: "https://custom.example.com/models/"},
			want: "https://custom.example.com/models",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := New(tc.opts)
			if got := p.baseURL(); got != tc.want {
				t.Errorf("baseURL() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestProvider_Model(t *testing.T) {
	p := New(Options{Project: "test-project", Location: "us-east5"})
	m := p.Model("claude-3-5-sonnet-v2@20241022")
	if m == nil {
		t.Fatal("Model() returned nil")
	}
	if got := m.ID(); got != "claude-3-5-sonnet-v2@20241022" {
		t.Errorf("Model().ID() = %q, want claude-3-5-sonnet-v2@20241022", got)
	}
	if got := m.Provider(); got != "vertex.anthropic.messages" {
		t.Errorf("Model().Provider() = %q, want vertex.anthropic.messages", got)
	}
}

func TestProvider_Tools(t *testing.T) {
	p := New(Options{Project: "test-project"})
	tools := p.Tools()
	wants := []string{
		"bash_20241022",
		"bash_20250124",
		"textEditor_20241022",
		"textEditor_20250124",
		"textEditor_20250429",
		"textEditor_20250728",
		"computer_20241022",
		"webSearch_20250305",
		"toolSearchRegex_20251119",
		"toolSearchBm25_20251119",
	}
	if len(tools) != len(wants) {
		t.Errorf("Tools() len = %d, want %d", len(tools), len(wants))
	}
	for _, w := range wants {
		if _, ok := tools[w]; !ok {
			t.Errorf("Tools() missing %q", w)
		}
	}
	// Ensure we don't leak Anthropic-only tools that Vertex doesn't accept.
	for _, forbidden := range []string{
		"codeExecution_20260120",
		"codeExecution_20250825",
		"webFetch_20260209",
		"webSearch_20260209",
		"computer_20250124",
		"computer_20251124",
	} {
		if _, ok := tools[forbidden]; ok {
			t.Errorf("Tools() should not expose %q", forbidden)
		}
	}
}

func TestProvider_StreamRequest(t *testing.T) {
	t.Run("builds URL with streamRawPredict suffix and strips model from body", func(t *testing.T) {
		var receivedPath string
		var receivedAuth string
		var receivedBody map[string]any
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedPath = r.URL.Path
			receivedAuth = r.Header.Get("Authorization")
			receivedHeaders = r.Header.Clone()
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			chunks := []string{
				`{"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
				`{"type":"message_stop"}`,
			}
			for _, c := range chunks {
				_, _ = w.Write([]byte("data: " + c + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		p := New(Options{
			Project:     "test-project",
			Location:    "us-east5",
			AccessToken: "bearer-token-123",
			BaseURL:     server.URL,
			Headers:     map[string]string{"X-Goog-User-Project": "test-project"},
		})
		m := p.Model("claude-3-5-sonnet-v2@20241022")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hello")},
		})
		if err != nil {
			t.Fatalf("Stream() error: %v", err)
		}
		for range events {
		}

		wantPath := "/claude-3-5-sonnet-v2@20241022:streamRawPredict"
		if receivedPath != wantPath {
			t.Errorf("request path = %q, want %q", receivedPath, wantPath)
		}
		if receivedAuth != "Bearer bearer-token-123" {
			t.Errorf("Authorization header = %q, want %q", receivedAuth, "Bearer bearer-token-123")
		}
		if got := receivedHeaders.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key header should not be set for bearer auth, got %q", got)
		}
		if got := receivedHeaders.Get("X-Goog-User-Project"); got != "test-project" {
			t.Errorf("custom header missing, got %q", got)
		}
		if _, ok := receivedBody["model"]; ok {
			t.Error("body.model should be stripped for Vertex Anthropic")
		}
		if got, _ := receivedBody["anthropic_version"].(string); got != "vertex-2023-10-16" {
			t.Errorf("body.anthropic_version = %q, want %q", got, "vertex-2023-10-16")
		}
	})
}

func TestProvider_StrictToolsStripped(t *testing.T) {
	// When a caller supplies a function tool, Vertex's SupportsStrictTools=false
	// must prevent any strict:true from showing up in the wire body.
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer server.Close()

	p := New(Options{
		Project:     "test-project",
		Location:    "us-east5",
		AccessToken: "token",
		BaseURL:     server.URL,
	})
	m := p.Model("claude-3-5-sonnet-v2@20241022")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		Tools: []tool.Tool{
			{
				Name:        "lookup",
				Description: "lookup",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	for range events {
	}

	tools, ok := receivedBody["tools"].([]any)
	if !ok || len(tools) == 0 {
		t.Fatalf("tools missing from body: %v", receivedBody)
	}
	for _, tt := range tools {
		if m, ok := tt.(map[string]any); ok {
			if _, has := m["strict"]; has {
				t.Errorf("tool entry should not have strict field: %v", m)
			}
		}
	}
}

func TestProvider_StructuredOutputFallsBackToSyntheticTool(t *testing.T) {
	// With SupportsNativeStructuredOutput=false the builder injects the
	// synthetic "json" tool rather than writing output_config.format.
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer server.Close()

	p := New(Options{
		Project:     "test-project",
		Location:    "us-east5",
		AccessToken: "token",
		BaseURL:     server.URL,
	})
	m := p.Model("claude-3-5-sonnet-v2@20241022")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("give me json")},
		ResponseFormat: &stream.ResponseFormat{
			Type:   "json",
			Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"}}}`),
		},
		ProviderOptions: map[string]any{
			"structuredOutputMode": "outputFormat",
		},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	for range events {
	}

	if _, ok := receivedBody["output_config"]; ok {
		t.Error("output_config should not appear when SupportsNativeStructuredOutput=false")
	}
	toolsRaw, ok := receivedBody["tools"].([]any)
	if !ok || len(toolsRaw) == 0 {
		t.Fatalf("expected synthetic 'json' tool in request body, got: %v", receivedBody)
	}
	sawJSONTool := false
	for _, tt := range toolsRaw {
		if m, ok := tt.(map[string]any); ok {
			if n, _ := m["name"].(string); n == "json" {
				sawJSONTool = true
				break
			}
		}
	}
	if !sawJSONTool {
		t.Errorf("synthetic 'json' tool not present: %v", toolsRaw)
	}
	// tool_choice should force the synthetic tool.
	tc, _ := receivedBody["tool_choice"].(map[string]any)
	if tc == nil || tc["name"] != "json" {
		t.Errorf("tool_choice should force name=json, got %v", tc)
	}
}

func TestProvider_SupportedURLsIsEmpty(t *testing.T) {
	p := New(Options{Project: "test-project"})
	cfg := p.config("claude-opus-4-7")
	if cfg.SupportedURLs == nil {
		t.Fatal("SupportedURLs should be set")
	}
	got := cfg.SupportedURLs()
	if len(got) != 0 {
		t.Errorf("SupportedURLs() = %v, want empty map", got)
	}
}

func TestProvider_BuildRequestURLGlobal(t *testing.T) {
	// Direct unit test of the URL builder hook for the global endpoint. The
	// constructed URL (p.baseURL() + ":rawPredict") must match ai-sdk's
	// global format.
	p := New(Options{Project: "test-project", Location: "global"})
	cfg := p.config("claude-opus-4-5@20251101")
	got := cfg.BuildRequestURL(p.baseURL(), false)
	want := "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/anthropic/models/claude-opus-4-5@20251101:rawPredict"
	if got != want {
		t.Errorf("BuildRequestURL global = %q, want %q", got, want)
	}
	got = cfg.BuildRequestURL(p.baseURL(), true)
	wantStream := strings.Replace(want, ":rawPredict", ":streamRawPredict", 1)
	if got != wantStream {
		t.Errorf("BuildRequestURL global stream = %q, want %q", got, wantStream)
	}
}
