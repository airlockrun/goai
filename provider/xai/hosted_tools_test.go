package xai

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for convertXaiProviderTool — translated from ai-sdk.
// Source: packages/xai/src/responses/xai-responses-prepare-tools.test.ts

func TestConvertXaiProviderTool_WebSearch(t *testing.T) {
	t.Run("defaults omit all optional fields", func(t *testing.T) {
		wire, ok := convertXaiProviderTool(WebSearch())
		if !ok {
			t.Fatal("expected ok=true for xai.web_search")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "web_search" {
			t.Errorf("type: got %v, want 'web_search'", got["type"])
		}
		for _, k := range []string{"allowed_domains", "excluded_domains", "enable_image_understanding"} {
			if _, present := got[k]; present {
				t.Errorf("%q should be omitted, got %v", k, got[k])
			}
		}
	})

	t.Run("populates all fields", func(t *testing.T) {
		enable := true
		wire, ok := convertXaiProviderTool(WebSearchWith(WebSearchOptions{
			AllowedDomains:           []string{"example.com", "other.dev"},
			ExcludedDomains:          []string{"blocked.com"},
			EnableImageUnderstanding: &enable,
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "web_search" {
			t.Errorf("type: got %v", got["type"])
		}
		allowed, _ := got["allowed_domains"].([]any)
		if len(allowed) != 2 || allowed[0] != "example.com" {
			t.Errorf("allowed_domains: got %v", allowed)
		}
		excluded, _ := got["excluded_domains"].([]any)
		if len(excluded) != 1 || excluded[0] != "blocked.com" {
			t.Errorf("excluded_domains: got %v", excluded)
		}
		if got["enable_image_understanding"] != true {
			t.Errorf("enable_image_understanding: got %v", got["enable_image_understanding"])
		}
	})
}

func TestConvertXaiProviderTool_XSearch(t *testing.T) {
	t.Run("defaults omit all optional fields", func(t *testing.T) {
		wire, ok := convertXaiProviderTool(XSearch())
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "x_search" {
			t.Errorf("type: got %v", got["type"])
		}
		for _, k := range []string{
			"allowed_x_handles", "excluded_x_handles", "from_date", "to_date",
			"enable_image_understanding", "enable_video_understanding",
		} {
			if _, present := got[k]; present {
				t.Errorf("%q should be omitted", k)
			}
		}
	})

	t.Run("populates all fields", func(t *testing.T) {
		enableImg := true
		enableVid := false
		wire, ok := convertXaiProviderTool(XSearchWith(XSearchOptions{
			AllowedXHandles:          []string{"elonmusk"},
			ExcludedXHandles:         []string{"spam"},
			FromDate:                 "2024-01-01",
			ToDate:                   "2025-01-01",
			EnableImageUnderstanding: &enableImg,
			EnableVideoUnderstanding: &enableVid,
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "x_search" {
			t.Errorf("type: got %v", got["type"])
		}
		if got["from_date"] != "2024-01-01" {
			t.Errorf("from_date: got %v", got["from_date"])
		}
		if got["to_date"] != "2025-01-01" {
			t.Errorf("to_date: got %v", got["to_date"])
		}
		if got["enable_image_understanding"] != true {
			t.Errorf("enable_image_understanding: got %v", got["enable_image_understanding"])
		}
		if got["enable_video_understanding"] != false {
			t.Errorf("enable_video_understanding: got %v", got["enable_video_understanding"])
		}
		if h, _ := got["allowed_x_handles"].([]any); len(h) != 1 || h[0] != "elonmusk" {
			t.Errorf("allowed_x_handles: got %v", got["allowed_x_handles"])
		}
		if h, _ := got["excluded_x_handles"].([]any); len(h) != 1 || h[0] != "spam" {
			t.Errorf("excluded_x_handles: got %v", got["excluded_x_handles"])
		}
	})
}

func TestConvertXaiProviderTool_CodeExecution(t *testing.T) {
	wire, ok := convertXaiProviderTool(CodeExecution())
	if !ok {
		t.Fatal("expected ok=true")
	}
	b, _ := json.Marshal(wire)
	var got map[string]any
	_ = json.Unmarshal(b, &got)
	// xai.code_execution maps to the API type "code_interpreter".
	if got["type"] != "code_interpreter" {
		t.Errorf("type: got %v, want 'code_interpreter'", got["type"])
	}
	if len(got) != 1 {
		t.Errorf("expected only type field, got %v", got)
	}
}

func TestConvertXaiProviderTool_ViewImage(t *testing.T) {
	wire, ok := convertXaiProviderTool(ViewImage())
	if !ok {
		t.Fatal("expected ok=true")
	}
	b, _ := json.Marshal(wire)
	var got map[string]any
	_ = json.Unmarshal(b, &got)
	if got["type"] != "view_image" {
		t.Errorf("type: got %v", got["type"])
	}
	if len(got) != 1 {
		t.Errorf("expected only type field, got %v", got)
	}
}

func TestConvertXaiProviderTool_ViewXVideo(t *testing.T) {
	wire, ok := convertXaiProviderTool(ViewXVideo())
	if !ok {
		t.Fatal("expected ok=true")
	}
	b, _ := json.Marshal(wire)
	var got map[string]any
	_ = json.Unmarshal(b, &got)
	if got["type"] != "view_x_video" {
		t.Errorf("type: got %v", got["type"])
	}
	if len(got) != 1 {
		t.Errorf("expected only type field, got %v", got)
	}
}

func TestConvertXaiProviderTool_FileSearch(t *testing.T) {
	t.Run("vector store IDs only", func(t *testing.T) {
		wire, ok := convertXaiProviderTool(FileSearch(FileSearchOptions{
			VectorStoreIDs: []string{"collection_1", "collection_2"},
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "file_search" {
			t.Errorf("type: got %v", got["type"])
		}
		ids, _ := got["vector_store_ids"].([]any)
		if len(ids) != 2 || ids[0] != "collection_1" || ids[1] != "collection_2" {
			t.Errorf("vector_store_ids: got %v", ids)
		}
		if _, present := got["max_num_results"]; present {
			t.Errorf("max_num_results should be omitted when unset")
		}
	})

	t.Run("with max num results", func(t *testing.T) {
		n := 10
		wire, ok := convertXaiProviderTool(FileSearch(FileSearchOptions{
			VectorStoreIDs: []string{"collection_1"},
			MaxNumResults:  &n,
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "file_search" {
			t.Errorf("type: got %v", got["type"])
		}
		if got["max_num_results"].(float64) != 10 {
			t.Errorf("max_num_results: got %v", got["max_num_results"])
		}
	})
}

func TestConvertXaiProviderTool_MCP(t *testing.T) {
	t.Run("minimal: serverUrl only", func(t *testing.T) {
		wire, ok := convertXaiProviderTool(MCP(MCPOptions{
			ServerURL: "https://example.com/mcp",
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "mcp" {
			t.Errorf("type: got %v", got["type"])
		}
		if got["server_url"] != "https://example.com/mcp" {
			t.Errorf("server_url: got %v", got["server_url"])
		}
		for _, k := range []string{"server_label", "server_description", "allowed_tools", "headers", "authorization"} {
			if _, present := got[k]; present {
				t.Errorf("%q should be omitted", k)
			}
		}
	})

	t.Run("populates all fields", func(t *testing.T) {
		wire, ok := convertXaiProviderTool(MCP(MCPOptions{
			ServerURL:         "https://example.com/mcp",
			ServerLabel:       "my-mcp",
			ServerDescription: "A test MCP server",
			AllowedTools:      []string{"search", "fetch"},
			Headers:           map[string]string{"X-API-Key": "secret"},
			Authorization:     "Bearer token",
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if got["type"] != "mcp" {
			t.Errorf("type: got %v", got["type"])
		}
		if got["server_url"] != "https://example.com/mcp" {
			t.Errorf("server_url: got %v", got["server_url"])
		}
		if got["server_label"] != "my-mcp" {
			t.Errorf("server_label: got %v", got["server_label"])
		}
		if got["server_description"] != "A test MCP server" {
			t.Errorf("server_description: got %v", got["server_description"])
		}
		if got["authorization"] != "Bearer token" {
			t.Errorf("authorization: got %v", got["authorization"])
		}
		tools, _ := got["allowed_tools"].([]any)
		if len(tools) != 2 || tools[0] != "search" || tools[1] != "fetch" {
			t.Errorf("allowed_tools: got %v", tools)
		}
		headers, _ := got["headers"].(map[string]any)
		if headers["X-API-Key"] != "secret" {
			t.Errorf("headers: got %v", headers)
		}
	})
}

func TestConvertXaiProviderTool_UnknownProviderReturnsFalse(t *testing.T) {
	wire, ok := convertXaiProviderTool(tool.Tool{
		Type:       "provider",
		ProviderID: "xai.not_a_real_tool",
		Args:       json.RawMessage("{}"),
	})
	if ok {
		t.Errorf("expected ok=false for unknown provider id, got wire=%v", wire)
	}
	if wire != nil {
		t.Errorf("expected nil wire, got %v", wire)
	}
}

func TestConvertToResponsesTools_IncludesProviderTools(t *testing.T) {
	// End-to-end: convertToResponsesTools must route provider-defined tools
	// through convertXaiProviderTool and emit the wire shape next to
	// function tools.
	tools := []tool.Tool{
		{
			Name:        "fn",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		},
		WebSearch(),
		XSearch(),
		CodeExecution(),
		ViewImage(),
		ViewXVideo(),
		FileSearch(FileSearchOptions{VectorStoreIDs: []string{"c1"}}),
		MCP(MCPOptions{ServerURL: "https://example.com/mcp"}),
	}

	result := convertToResponsesTools(tools)
	if len(result) != 8 {
		t.Fatalf("expected 8 tools, got %d", len(result))
	}

	types := make([]string, 0, len(result))
	for _, item := range result {
		b, _ := json.Marshal(item)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		s, _ := m["type"].(string)
		types = append(types, s)
	}

	expected := []string{
		"function", "web_search", "x_search", "code_interpreter",
		"view_image", "view_x_video", "file_search", "mcp",
	}
	for i, want := range expected {
		if types[i] != want {
			t.Errorf("tools[%d].type: got %q, want %q", i, types[i], want)
		}
	}
}

func TestConvertToResponsesTools_SkipsUnknownProviderTools(t *testing.T) {
	tools := []tool.Tool{
		{
			Name:        "fn",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		},
		{
			Type:       "provider",
			ProviderID: "xai.not_real",
			Args:       json.RawMessage("{}"),
		},
	}
	result := convertToResponsesTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool (unknown provider skipped), got %d", len(result))
	}
	if _, ok := result[0].(responsesTool); !ok {
		t.Errorf("expected responsesTool, got %T", result[0])
	}
}

// Factory wrapper assertions — mirror ai-sdk callers that check
// Type/ProviderID/Args shape without going through the wire converter.
func TestHostedToolFactories_ProducedTools(t *testing.T) {
	cases := []struct {
		name   string
		tool   tool.Tool
		wantID string
	}{
		{"WebSearch", WebSearch(), ToolIDWebSearch},
		{"XSearch", XSearch(), ToolIDXSearch},
		{"CodeExecution", CodeExecution(), ToolIDCodeExecution},
		{"ViewImage", ViewImage(), ToolIDViewImage},
		{"ViewXVideo", ViewXVideo(), ToolIDViewXVideo},
		{"FileSearch", FileSearch(FileSearchOptions{VectorStoreIDs: []string{"c1"}}), ToolIDFileSearch},
		{"MCP", MCP(MCPOptions{ServerURL: "u"}), ToolIDMCP},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.tool.Type != "provider" {
				t.Errorf("Type: got %q, want 'provider'", tc.tool.Type)
			}
			if tc.tool.ProviderID != tc.wantID {
				t.Errorf("ProviderID: got %q, want %q", tc.tool.ProviderID, tc.wantID)
			}
			if len(tc.tool.Args) == 0 {
				t.Errorf("Args should not be empty")
			}
		})
	}
}

// Tests for normalizeResponsesToolChoice — translated from ai-sdk
// commit 05f3f36 (toolChoice warning for server-side hosted tools).

func TestNormalizeResponsesToolChoice_StringsPassThrough(t *testing.T) {
	for _, v := range []string{"auto", "none", "required"} {
		got := normalizeResponsesToolChoice(v, nil)
		if got != v {
			t.Errorf("%q: got %v, want %q", v, got, v)
		}
	}
}

func TestNormalizeResponsesToolChoice_ForceFunctionTool(t *testing.T) {
	// V3-shaped tool choice for a plain function tool is translated
	// to xAI's {"type": "function", "name": "..."} wire shape.
	tools := []tool.Tool{
		{
			Name:        "calculator",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		},
	}
	got := normalizeResponsesToolChoice(map[string]any{
		"type":     "tool",
		"toolName": "calculator",
	}, tools)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T", got)
	}
	if m["type"] != "function" || m["name"] != "calculator" {
		t.Errorf("got %v, want {type: function, name: calculator}", m)
	}
}

func TestNormalizeResponsesToolChoice_DropsForcedHostedTools(t *testing.T) {
	// xAI rejects a forced server-side hosted tool; goai drops the choice.
	cases := []struct {
		name string
		tool tool.Tool
	}{
		{"web_search", WebSearch()},
		{"x_search", XSearch()},
		{"code_execution", CodeExecution()},
		{"view_image", ViewImage()},
		{"view_x_video", ViewXVideo()},
		{"file_search", FileSearch(FileSearchOptions{VectorStoreIDs: []string{"c1"}})},
		{"mcp", MCP(MCPOptions{ServerURL: "https://example.com/mcp"})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tools := []tool.Tool{tc.tool}
			got := normalizeResponsesToolChoice(map[string]any{
				"type":     "tool",
				"toolName": tc.tool.ProviderID,
			}, tools)
			if got != nil {
				t.Errorf("expected nil (dropped), got %v", got)
			}
		})
	}
}

func TestNormalizeResponsesToolChoice_UnknownToolPassesThroughAsFunction(t *testing.T) {
	// If the named tool isn't in the tools slice, we still translate the
	// V3 shape to the xAI function shape — matches ai-sdk behavior which
	// treats unresolved lookups as function references.
	got := normalizeResponsesToolChoice(map[string]any{
		"type":     "tool",
		"toolName": "unknown_tool",
	}, nil)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T", got)
	}
	if m["type"] != "function" || m["name"] != "unknown_tool" {
		t.Errorf("got %v", m)
	}
}

func TestNormalizeResponsesToolChoice_NonToolMapPassesThrough(t *testing.T) {
	// Callers passing xAI-native shapes {"type": "function", "name": "..."}
	// are left alone.
	in := map[string]any{"type": "function", "name": "foo"}
	got := normalizeResponsesToolChoice(in, nil)
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T", got)
	}
	if m["type"] != "function" || m["name"] != "foo" {
		t.Errorf("got %v", m)
	}
}
