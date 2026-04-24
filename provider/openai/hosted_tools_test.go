package openai

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests for convertOpenAIProviderTool — translated from ai-sdk.
// Source: packages/openai/src/responses/openai-responses-prepare-tools.test.ts

func TestConvertOpenAIProviderTool_WebSearch(t *testing.T) {
	t.Run("defaults omit all optional fields", func(t *testing.T) {
		wire, ok := convertOpenAIProviderTool(WebSearch())
		if !ok {
			t.Fatal("expected ok=true for openai.web_search")
		}

		b, err := json.Marshal(wire)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var got map[string]any
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got["type"] != "web_search" {
			t.Errorf("expected type 'web_search', got %v", got["type"])
		}
		for _, k := range []string{"external_web_access", "filters", "search_context_size", "user_location"} {
			if _, present := got[k]; present {
				t.Errorf("expected %q to be omitted, got %v", k, got[k])
			}
		}
	})

	t.Run("populates all fields", func(t *testing.T) {
		externalAccess := true
		wire, ok := convertOpenAIProviderTool(WebSearchWith(WebSearchOptions{
			ExternalWebAccess: &externalAccess,
			Filters:           &WebSearchFilters{AllowedDomains: []string{"example.com", "other.dev"}},
			SearchContextSize: "high",
			UserLocation: &WebSearchUserLocation{
				Type:     "approximate",
				Country:  "US",
				City:     "San Francisco",
				Region:   "California",
				Timezone: "America/Los_Angeles",
			},
		}))
		if !ok {
			t.Fatal("expected ok=true")
		}

		b, err := json.Marshal(wire)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}

		var got map[string]any
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if got["type"] != "web_search" {
			t.Errorf("type: got %v, want 'web_search'", got["type"])
		}
		if got["external_web_access"] != true {
			t.Errorf("external_web_access: got %v, want true", got["external_web_access"])
		}
		if got["search_context_size"] != "high" {
			t.Errorf("search_context_size: got %v, want 'high'", got["search_context_size"])
		}
		filters, ok := got["filters"].(map[string]any)
		if !ok {
			t.Fatalf("filters missing or wrong type: %v", got["filters"])
		}
		domains, ok := filters["allowed_domains"].([]any)
		if !ok || len(domains) != 2 {
			t.Errorf("allowed_domains: got %v", domains)
		}
		loc, ok := got["user_location"].(map[string]any)
		if !ok {
			t.Fatalf("user_location missing: %v", got["user_location"])
		}
		if loc["country"] != "US" || loc["city"] != "San Francisco" || loc["timezone"] != "America/Los_Angeles" {
			t.Errorf("user_location fields wrong: %v", loc)
		}
	})

	t.Run("filters-only serialization", func(t *testing.T) {
		wire, _ := convertOpenAIProviderTool(WebSearchWith(WebSearchOptions{
			Filters: &WebSearchFilters{AllowedDomains: []string{"example.com"}},
		}))
		b, _ := json.Marshal(wire)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		if _, present := got["external_web_access"]; present {
			t.Errorf("external_web_access should be omitted")
		}
		filters, ok := got["filters"].(map[string]any)
		if !ok {
			t.Fatalf("filters missing")
		}
		if filters["allowed_domains"].([]any)[0] != "example.com" {
			t.Errorf("allowed_domains wrong: %v", filters["allowed_domains"])
		}
	})
}

func TestConvertOpenAIProviderTool_Custom(t *testing.T) {
	cases := []struct {
		name    string
		opts    CustomOptions
		wantFmt map[string]any
	}{
		{
			name: "name only omits format and description",
			opts: CustomOptions{Name: "write_sql"},
		},
		{
			name: "grammar format regex",
			opts: CustomOptions{
				Name:        "write_sql",
				Description: "Executes SQL",
				Format: &CustomFormat{
					Type:       "grammar",
					Syntax:     "regex",
					Definition: "^SELECT .+$",
				},
			},
			wantFmt: map[string]any{
				"type":       "grammar",
				"syntax":     "regex",
				"definition": "^SELECT .+$",
			},
		},
		{
			name: "grammar format lark",
			opts: CustomOptions{
				Name: "write_sql",
				Format: &CustomFormat{
					Type:       "grammar",
					Syntax:     "lark",
					Definition: "start: \"SELECT\"",
				},
			},
			wantFmt: map[string]any{
				"type":       "grammar",
				"syntax":     "lark",
				"definition": "start: \"SELECT\"",
			},
		},
		{
			name: "text format",
			opts: CustomOptions{
				Name:   "write_sql",
				Format: &CustomFormat{Type: "text"},
			},
			wantFmt: map[string]any{"type": "text"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wire, ok := convertOpenAIProviderTool(Custom(tc.opts))
			if !ok {
				t.Fatal("expected ok=true")
			}
			b, err := json.Marshal(wire)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got["type"] != "custom" {
				t.Errorf("type: got %v, want 'custom'", got["type"])
			}
			if got["name"] != tc.opts.Name {
				t.Errorf("name: got %v, want %q", got["name"], tc.opts.Name)
			}
			if tc.opts.Description == "" {
				if _, present := got["description"]; present {
					t.Errorf("description should be omitted")
				}
			} else if got["description"] != tc.opts.Description {
				t.Errorf("description: got %v, want %q", got["description"], tc.opts.Description)
			}
			if tc.wantFmt == nil {
				if _, present := got["format"]; present {
					t.Errorf("format should be omitted")
				}
				return
			}
			format, ok := got["format"].(map[string]any)
			if !ok {
				t.Fatalf("format missing: %v", got["format"])
			}
			for k, v := range tc.wantFmt {
				if format[k] != v {
					t.Errorf("format[%q]: got %v, want %v", k, format[k], v)
				}
			}
		})
	}
}

func TestCustom_PreservesAliasName(t *testing.T) {
	// ai-sdk 58bc42d: the goai-side tool.Name (surface/alias) must equal
	// the wire-format name so downstream lookup resolves correctly.
	tl := Custom(CustomOptions{Name: "write_sql"})
	if tl.Name != "write_sql" {
		t.Errorf("tool.Name: got %q, want %q", tl.Name, "write_sql")
	}
	if tl.Type != "provider" {
		t.Errorf("tool.Type: got %q, want 'provider'", tl.Type)
	}
	if tl.ProviderID != ToolIDCustom {
		t.Errorf("tool.ProviderID: got %q, want %q", tl.ProviderID, ToolIDCustom)
	}
}

func TestConvertOpenAIProviderTool_ToolSearch(t *testing.T) {
	cases := []struct {
		name string
		opts ToolSearchOptions
		want map[string]any
	}{
		{
			name: "defaults omit execution/description/parameters",
			opts: ToolSearchOptions{},
			want: map[string]any{"type": "tool_search"},
		},
		{
			name: "server execution",
			opts: ToolSearchOptions{Execution: "server"},
			want: map[string]any{"type": "tool_search", "execution": "server"},
		},
		{
			name: "client execution with description and parameters",
			opts: ToolSearchOptions{
				Execution:   "client",
				Description: "Search custom tools",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"q": map[string]any{"type": "string"}},
				},
			},
			want: map[string]any{
				"type":        "tool_search",
				"execution":   "client",
				"description": "Search custom tools",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"q": map[string]any{"type": "string"}},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wire, ok := convertOpenAIProviderTool(ToolSearch(tc.opts))
			if !ok {
				t.Fatal("expected ok=true")
			}
			b, err := json.Marshal(wire)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}

			for k, v := range tc.want {
				gotV := got[k]
				vb, _ := json.Marshal(v)
				gb, _ := json.Marshal(gotV)
				if string(vb) != string(gb) {
					t.Errorf("key %q: got %s, want %s", k, gb, vb)
				}
			}

			// Verify nothing extra leaks in.
			for k := range got {
				if _, expected := tc.want[k]; !expected {
					t.Errorf("unexpected key %q in output: %v", k, got[k])
				}
			}
		})
	}
}

func TestConvertOpenAIProviderTool_UnknownProviderReturnsFalse(t *testing.T) {
	wire, ok := convertOpenAIProviderTool(tool.Tool{
		Type:       "provider",
		ProviderID: "openai.not_a_real_tool",
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
	// through convertOpenAIProviderTool and emit the wire shape next to
	// function tools.
	tools := []tool.Tool{
		{
			Name:        "fn",
			Description: "",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		},
		WebSearch(),
		Custom(CustomOptions{Name: "write_sql"}),
		ToolSearch(ToolSearchOptions{Execution: "server"}),
	}

	result := convertToResponsesTools(tools, false)
	if len(result) != 4 {
		t.Fatalf("expected 4 tools, got %d", len(result))
	}

	types := make([]string, 0, len(result))
	for _, item := range result {
		b, _ := json.Marshal(item)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		t, _ := m["type"].(string)
		types = append(types, t)
	}

	expected := []string{"function", "web_search", "custom", "tool_search"}
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
			ProviderID: "openai.not_real",
			Args:       json.RawMessage("{}"),
		},
	}
	result := convertToResponsesTools(tools, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool (unknown provider skipped), got %d", len(result))
	}
	if _, ok := result[0].(responsesTool); !ok {
		t.Errorf("expected responsesTool, got %T", result[0])
	}
}

func TestConvertToChatTools_SkipsProviderTools(t *testing.T) {
	// Chat Completions doesn't support provider-defined hosted tools;
	// ai-sdk's prepareChatTools emits an 'unsupported' warning. goai
	// silently drops them.
	tools := []tool.Tool{
		{
			Name:        "fn",
			InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
		},
		WebSearch(),
		Custom(CustomOptions{Name: "write_sql"}),
		ToolSearch(ToolSearchOptions{}),
	}
	result := convertToChatTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool (provider tools skipped), got %d", len(result))
	}
	if result[0].Function.Name != "fn" {
		t.Errorf("expected function name 'fn', got %q", result[0].Function.Name)
	}
}
