package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// Tests translated from ai-sdk/packages/anthropic/src/anthropic-prepare-tools.test.ts
// coverage for the provider-defined hosted tools.

func TestHostedTool_CodeExecution20260120_NoBeta(t *testing.T) {
	result, betas := convertToAnthropicTools([]tool.Tool{CodeExecution()})

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	got := result[0].(anthropicHostedCodeExecution20260120)
	if got.Type != "code_execution_20260120" || got.Name != "code_execution" {
		t.Errorf("wire shape mismatch: %+v", got)
	}
	// code_execution_20260120 is explicitly the version that does NOT
	// require a beta header.
	if len(betas) != 0 {
		t.Errorf("expected no beta headers, got %v", betas)
	}
}

func TestHostedTool_WebSearch20260209_AnnouncesBeta(t *testing.T) {
	tools := []tool.Tool{
		WebSearchWith(WebSearchOptions{
			MaxUses:        5,
			AllowedDomains: []string{"example.com"},
			UserLocation: &WebSearchUserLocale{
				Type:    "approximate",
				City:    "Paris",
				Country: "FR",
			},
		}),
	}
	result, betas := convertToAnthropicTools(tools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	got := result[0].(anthropicHostedWebSearch20260209)
	if got.Type != "web_search_20260209" || got.Name != "web_search" {
		t.Errorf("wire shape mismatch: %+v", got)
	}
	if got.MaxUses != 5 {
		t.Errorf("MaxUses = %d, want 5", got.MaxUses)
	}
	if got.UserLocation == nil || got.UserLocation.City != "Paris" {
		t.Errorf("UserLocation = %+v", got.UserLocation)
	}
	// Wire JSON must use snake_case keys.
	wire, _ := json.Marshal(got)
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if _, ok := decoded["max_uses"]; !ok {
		t.Errorf("expected max_uses key in wire payload: %s", wire)
	}
	if _, ok := decoded["user_location"]; !ok {
		t.Errorf("expected user_location key in wire payload: %s", wire)
	}
	if len(betas) != 1 || betas[0] != "code-execution-web-tools-2026-02-09" {
		t.Errorf("expected code-execution-web-tools beta, got %v", betas)
	}
}

func TestHostedTool_WebFetch20260209_Options(t *testing.T) {
	tools := []tool.Tool{
		WebFetchWith(WebFetchOptions{MaxContentTokens: 10000}),
	}
	result, betas := convertToAnthropicTools(tools)

	got := result[0].(anthropicHostedWebFetch20260209)
	if got.Type != "web_fetch_20260209" || got.Name != "web_fetch" {
		t.Errorf("wire shape mismatch: %+v", got)
	}
	if got.MaxContentTokens != 10000 {
		t.Errorf("MaxContentTokens = %d, want 10000", got.MaxContentTokens)
	}
	if len(betas) != 1 {
		t.Errorf("expected 1 beta, got %v", betas)
	}
}

func TestHostedTool_ToolSearch_WireShape(t *testing.T) {
	tests := []struct {
		name     string
		tool     tool.Tool
		wantType string
		wantName string
	}{
		{
			name:     "regex",
			tool:     ToolSearchRegex(),
			wantType: "tool_search_tool_regex_20251119",
			wantName: "tool_search_tool_regex",
		},
		{
			name:     "bm25",
			tool:     ToolSearchBM25(),
			wantType: "tool_search_tool_bm25_20251119",
			wantName: "tool_search_tool_bm25",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, betas := convertToAnthropicTools([]tool.Tool{tc.tool})
			if len(result) != 1 {
				t.Fatalf("expected 1 tool, got %d", len(result))
			}
			wire, err := json.Marshal(result[0])
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var decoded map[string]any
			if err := json.Unmarshal(wire, &decoded); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if decoded["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", decoded["type"], tc.wantType)
			}
			if decoded["name"] != tc.wantName {
				t.Errorf("name = %v, want %q", decoded["name"], tc.wantName)
			}
			if len(decoded) != 2 {
				t.Errorf("expected exactly 2 fields (type, name), got %d: %v", len(decoded), decoded)
			}
			// tool-search is GA on the direct Anthropic API — no beta
			// header required. Bedrock's beta requirement is handled in
			// the bedrock provider.
			if len(betas) != 0 {
				t.Errorf("expected no beta headers, got %v", betas)
			}
		})
	}
}

func TestHostedTool_MixedFunctionAndHosted(t *testing.T) {
	tools := []tool.Tool{
		{
			Name:        "my_fn",
			Description: "A function tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
		WebSearch(),
	}
	result, betas := convertToAnthropicTools(tools)

	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if _, ok := result[0].(anthropicTool); !ok {
		t.Errorf("result[0] should be function tool, got %T", result[0])
	}
	if _, ok := result[1].(anthropicHostedWebSearch20260209); !ok {
		t.Errorf("result[1] should be hosted web_search, got %T", result[1])
	}
	if len(betas) != 1 {
		t.Errorf("expected 1 beta from hosted tool, got %v", betas)
	}
}

// Wire-shape + beta coverage for the tools ported as part of the
// ai-sdk@6.0.168 follow-up (bash/text_editor/computer/web_search_20250305).
// Mirrors ai-sdk/packages/anthropic/src/anthropic-prepare-tools.ts.

func TestHostedTool_SimpleWireShape(t *testing.T) {
	tests := []struct {
		name     string
		tool     tool.Tool
		wantType string
		wantName string
		wantBeta string
	}{
		{"bash_20241022", Bash20241022(), "bash_20241022", "bash", "computer-use-2024-10-22"},
		{"bash_20250124", Bash20250124(), "bash_20250124", "bash", "computer-use-2025-01-24"},
		{"text_editor_20241022", TextEditor20241022(), "text_editor_20241022", "str_replace_editor", "computer-use-2024-10-22"},
		{"text_editor_20250124", TextEditor20250124(), "text_editor_20250124", "str_replace_editor", "computer-use-2025-01-24"},
		{"text_editor_20250429", TextEditor20250429(), "text_editor_20250429", "str_replace_based_edit_tool", "computer-use-2025-01-24"},
		{"text_editor_20250728", TextEditor20250728(), "text_editor_20250728", "str_replace_based_edit_tool", ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, betas := convertToAnthropicTools([]tool.Tool{tc.tool})
			if len(result) != 1 {
				t.Fatalf("expected 1 tool, got %d", len(result))
			}
			wire, err := json.Marshal(result[0])
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var decoded map[string]any
			if err := json.Unmarshal(wire, &decoded); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if decoded["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", decoded["type"], tc.wantType)
			}
			if decoded["name"] != tc.wantName {
				t.Errorf("name = %v, want %q", decoded["name"], tc.wantName)
			}
			if tc.wantBeta == "" {
				if len(betas) != 0 {
					t.Errorf("expected no betas, got %v", betas)
				}
			} else {
				if len(betas) != 1 || betas[0] != tc.wantBeta {
					t.Errorf("betas = %v, want [%q]", betas, tc.wantBeta)
				}
			}
		})
	}
}

func TestHostedTool_TextEditor20250728_MaxCharacters(t *testing.T) {
	result, betas := convertToAnthropicTools([]tool.Tool{
		TextEditor20250728With(TextEditorOptions{MaxCharacters: 5000}),
	})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	got := result[0].(anthropicHostedTextEditor20250728)
	if got.MaxCharacters != 5000 {
		t.Errorf("MaxCharacters = %d, want 5000", got.MaxCharacters)
	}
	wire, _ := json.Marshal(got)
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if decoded["max_characters"] != float64(5000) {
		t.Errorf("wire max_characters = %v, want 5000", decoded["max_characters"])
	}
	if len(betas) != 0 {
		t.Errorf("expected no betas, got %v", betas)
	}
}

func TestHostedTool_TextEditor20250728_OmitsEmptyMaxCharacters(t *testing.T) {
	result, _ := convertToAnthropicTools([]tool.Tool{TextEditor20250728()})
	wire, _ := json.Marshal(result[0])
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if _, has := decoded["max_characters"]; has {
		t.Errorf("max_characters should be omitted when zero, got %v", decoded)
	}
}

func TestHostedTool_ComputerWireShape(t *testing.T) {
	displayNo := 2
	opts := ComputerOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		DisplayNumber:   &displayNo,
	}

	tests := []struct {
		name     string
		tool     tool.Tool
		wantType string
		wantBeta string
	}{
		{"computer_20241022", Computer20241022With(opts), "computer_20241022", "computer-use-2024-10-22"},
		{"computer_20250124", Computer20250124With(opts), "computer_20250124", "computer-use-2025-01-24"},
		{"computer_20251124", Computer20251124With(opts), "computer_20251124", "computer-use-2025-11-24"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, betas := convertToAnthropicTools([]tool.Tool{tc.tool})
			if len(result) != 1 {
				t.Fatalf("expected 1 tool, got %d", len(result))
			}
			wire, err := json.Marshal(result[0])
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var decoded map[string]any
			if err := json.Unmarshal(wire, &decoded); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if decoded["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", decoded["type"], tc.wantType)
			}
			if decoded["name"] != "computer" {
				t.Errorf("name = %v, want computer", decoded["name"])
			}
			if decoded["display_width_px"] != float64(1920) {
				t.Errorf("display_width_px = %v, want 1920", decoded["display_width_px"])
			}
			if decoded["display_height_px"] != float64(1080) {
				t.Errorf("display_height_px = %v, want 1080", decoded["display_height_px"])
			}
			if decoded["display_number"] != float64(2) {
				t.Errorf("display_number = %v, want 2", decoded["display_number"])
			}
			if len(betas) != 1 || betas[0] != tc.wantBeta {
				t.Errorf("betas = %v, want [%q]", betas, tc.wantBeta)
			}
		})
	}
}

func TestHostedTool_Computer20251124_EnableZoom(t *testing.T) {
	zoom := true
	result, _ := convertToAnthropicTools([]tool.Tool{
		Computer20251124With(ComputerOptions{
			DisplayWidthPx:  800,
			DisplayHeightPx: 600,
			EnableZoom:      &zoom,
		}),
	})
	wire, _ := json.Marshal(result[0])
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if decoded["enable_zoom"] != true {
		t.Errorf("enable_zoom = %v, want true", decoded["enable_zoom"])
	}
}

func TestHostedTool_Computer_OmitsNilDisplayNumber(t *testing.T) {
	result, _ := convertToAnthropicTools([]tool.Tool{
		Computer20241022With(ComputerOptions{DisplayWidthPx: 800, DisplayHeightPx: 600}),
	})
	wire, _ := json.Marshal(result[0])
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if _, has := decoded["display_number"]; has {
		t.Errorf("display_number should be omitted when nil, got %v", decoded)
	}
	if _, has := decoded["enable_zoom"]; has {
		t.Errorf("enable_zoom should be omitted on computer_20241022, got %v", decoded)
	}
}

func TestHostedTool_WebSearch20250305_Options(t *testing.T) {
	tools := []tool.Tool{
		WebSearch20250305With(WebSearchOptions{
			MaxUses:        3,
			AllowedDomains: []string{"example.com"},
			BlockedDomains: []string{"spam.example"},
			UserLocation: &WebSearchUserLocale{
				Type:    "approximate",
				City:    "Berlin",
				Country: "DE",
			},
		}),
	}
	result, betas := convertToAnthropicTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	got := result[0].(anthropicHostedWebSearch20250305)
	if got.Type != "web_search_20250305" || got.Name != "web_search" {
		t.Errorf("wire shape mismatch: %+v", got)
	}
	if got.MaxUses != 3 {
		t.Errorf("MaxUses = %d, want 3", got.MaxUses)
	}
	if got.UserLocation == nil || got.UserLocation.City != "Berlin" {
		t.Errorf("UserLocation = %+v", got.UserLocation)
	}
	wire, _ := json.Marshal(got)
	var decoded map[string]any
	_ = json.Unmarshal(wire, &decoded)
	if decoded["max_uses"] != float64(3) {
		t.Errorf("wire max_uses = %v", decoded["max_uses"])
	}
	if _, ok := decoded["user_location"]; !ok {
		t.Errorf("wire missing user_location: %s", wire)
	}
	// web_search_20250305 predates the code-execution-web-tools beta.
	if len(betas) != 0 {
		t.Errorf("expected no betas, got %v", betas)
	}
}

func TestHostedTool_WebSearch20260209_StillAnnouncesBeta(t *testing.T) {
	// Regression: sharing WebSearchOptions between 20250305 and 20260209
	// must not regress the 20260209 beta-header behaviour.
	_, betas := convertToAnthropicTools([]tool.Tool{WebSearch()})
	if len(betas) != 1 || betas[0] != "code-execution-web-tools-2026-02-09" {
		t.Errorf("expected code-execution-web-tools beta, got %v", betas)
	}
}

func TestConvertProviderTool_BetaHeaders(t *testing.T) {
	tests := []struct {
		id       string
		builder  func() tool.Tool
		wantBeta string
	}{
		{ToolIDBash20241022, Bash20241022, "computer-use-2024-10-22"},
		{ToolIDBash20250124, Bash20250124, "computer-use-2025-01-24"},
		{ToolIDTextEditor20241022, TextEditor20241022, "computer-use-2024-10-22"},
		{ToolIDTextEditor20250124, TextEditor20250124, "computer-use-2025-01-24"},
		{ToolIDTextEditor20250429, TextEditor20250429, "computer-use-2025-01-24"},
		{ToolIDTextEditor20250728, TextEditor20250728, ""},
		{ToolIDWebSearch20250305, WebSearch20250305, ""},
		{ToolIDComputer20241022, func() tool.Tool {
			return Computer20241022With(ComputerOptions{DisplayWidthPx: 1, DisplayHeightPx: 1})
		}, "computer-use-2024-10-22"},
		{ToolIDComputer20250124, func() tool.Tool {
			return Computer20250124With(ComputerOptions{DisplayWidthPx: 1, DisplayHeightPx: 1})
		}, "computer-use-2025-01-24"},
		{ToolIDComputer20251124, func() tool.Tool {
			return Computer20251124With(ComputerOptions{DisplayWidthPx: 1, DisplayHeightPx: 1})
		}, "computer-use-2025-11-24"},
	}

	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			_, beta, ok := ConvertProviderTool(tc.builder())
			if !ok {
				t.Fatalf("ConvertProviderTool returned ok=false for %q", tc.id)
			}
			if beta != tc.wantBeta {
				t.Errorf("beta = %q, want %q", beta, tc.wantBeta)
			}
		})
	}
}
