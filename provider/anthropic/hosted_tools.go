package anthropic

import (
	"encoding/json"

	"github.com/airlockrun/goai/tool"
)

// Provider-defined tool IDs for Anthropic hosted tools. Mirror ai-sdk's
// factory IDs at packages/anthropic/src/tool/*.ts.
const (
	ToolIDCodeExecution20260120   = "anthropic.code_execution_20260120"
	ToolIDCodeExecution20250825   = "anthropic.code_execution_20250825"
	ToolIDWebSearch20260209       = "anthropic.web_search_20260209"
	ToolIDWebSearch20250305       = "anthropic.web_search_20250305"
	ToolIDWebFetch20260209        = "anthropic.web_fetch_20260209"
	ToolIDToolSearchRegex20251119 = "anthropic.tool_search_regex_20251119"
	ToolIDToolSearchBM25_20251119 = "anthropic.tool_search_bm25_20251119"
	ToolIDBash20241022            = "anthropic.bash_20241022"
	ToolIDBash20250124            = "anthropic.bash_20250124"
	ToolIDTextEditor20241022      = "anthropic.text_editor_20241022"
	ToolIDTextEditor20250124      = "anthropic.text_editor_20250124"
	ToolIDTextEditor20250429      = "anthropic.text_editor_20250429"
	ToolIDTextEditor20250728      = "anthropic.text_editor_20250728"
	ToolIDComputer20241022        = "anthropic.computer_20241022"
	ToolIDComputer20250124        = "anthropic.computer_20250124"
	ToolIDComputer20251124        = "anthropic.computer_20251124"
	ToolIDAdvisor20260301         = "anthropic.advisor_20260301"
)

// CodeExecution returns a provider-defined tool that enables Anthropic's
// server-side code execution (Python + Bash, recommended version — no
// beta header required, supported on Opus 4.6/4.5, Sonnet 4.6/4.5).
func CodeExecution() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDCodeExecution20260120,
		Args:       json.RawMessage("{}"),
	}
}

// WebSearchOptions configures the anthropic.web_search_20260209 hosted
// tool. Mirrors the ai-sdk argsSchema at
// packages/anthropic/src/tool/web-search_20260209.ts.
type WebSearchOptions struct {
	MaxUses        int                  `json:"maxUses,omitempty"`
	AllowedDomains []string             `json:"allowedDomains,omitempty"`
	BlockedDomains []string             `json:"blockedDomains,omitempty"`
	UserLocation   *WebSearchUserLocale `json:"userLocation,omitempty"`
}

// WebSearchUserLocale narrows results to a geography. Type is always
// "approximate".
type WebSearchUserLocale struct {
	Type     string `json:"type"` // "approximate"
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearch returns a provider-defined Anthropic web_search_20260209
// tool. Use WebSearchWith for options.
func WebSearch() tool.Tool { return WebSearchWith(WebSearchOptions{}) }

func WebSearchWith(opts WebSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDWebSearch20260209,
		Args:       args,
	}
}

// WebFetchOptions configures the anthropic.web_fetch_20260209 hosted
// tool. Mirrors the ai-sdk argsSchema at
// packages/anthropic/src/tool/web-fetch-20260209.ts.
type WebFetchOptions struct {
	MaxUses          int      `json:"maxUses,omitempty"`
	AllowedDomains   []string `json:"allowedDomains,omitempty"`
	BlockedDomains   []string `json:"blockedDomains,omitempty"`
	Citations        any      `json:"citations,omitempty"`
	MaxContentTokens int      `json:"maxContentTokens,omitempty"`
}

// WebFetch returns a provider-defined Anthropic web_fetch_20260209 tool.
func WebFetch() tool.Tool { return WebFetchWith(WebFetchOptions{}) }

func WebFetchWith(opts WebFetchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDWebFetch20260209,
		Args:       args,
	}
}

// ToolSearchRegex returns a provider-defined Anthropic
// tool_search_regex_20251119 tool. The tool search tool enables Claude
// to work with large tool catalogs by dynamically discovering and
// loading tools on-demand using Python re.search()-style regex
// patterns.
func ToolSearchRegex() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDToolSearchRegex20251119,
		Args:       json.RawMessage("{}"),
	}
}

// ToolSearchBM25 returns a provider-defined Anthropic
// tool_search_bm25_20251119 tool. Uses BM25 natural-language queries
// to discover tools from a large catalog on-demand.
func ToolSearchBM25() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDToolSearchBM25_20251119,
		Args:       json.RawMessage("{}"),
	}
}

// Bash20241022 returns a provider-defined Anthropic bash_20241022 tool
// (requires the computer-use-2024-10-22 beta on the direct Anthropic API).
func Bash20241022() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDBash20241022,
		Args:       json.RawMessage("{}"),
	}
}

// Bash20250124 returns a provider-defined Anthropic bash_20250124 tool
// (requires the computer-use-2025-01-24 beta on the direct Anthropic API).
func Bash20250124() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDBash20250124,
		Args:       json.RawMessage("{}"),
	}
}

// TextEditor20241022 returns the str_replace_editor-flavored text editor
// tool (requires computer-use-2024-10-22 beta).
func TextEditor20241022() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDTextEditor20241022,
		Args:       json.RawMessage("{}"),
	}
}

// TextEditor20250124 returns the str_replace_editor-flavored text editor
// tool (requires computer-use-2025-01-24 beta).
func TextEditor20250124() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDTextEditor20250124,
		Args:       json.RawMessage("{}"),
	}
}

// TextEditor20250429 returns the str_replace_based_edit_tool-flavored text
// editor tool (requires computer-use-2025-01-24 beta).
func TextEditor20250429() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDTextEditor20250429,
		Args:       json.RawMessage("{}"),
	}
}

// TextEditorOptions configures the anthropic.text_editor_20250728 hosted
// tool. Mirrors the ai-sdk argsSchema at
// packages/anthropic/src/tool/text-editor_20250728.ts.
type TextEditorOptions struct {
	MaxCharacters int `json:"maxCharacters,omitempty"`
}

// TextEditor20250728 returns the str_replace_based_edit_tool text-editor
// tool. Use TextEditor20250728With for options.
func TextEditor20250728() tool.Tool { return TextEditor20250728With(TextEditorOptions{}) }

// TextEditor20250728With returns the tool with explicit options.
func TextEditor20250728With(opts TextEditorOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDTextEditor20250728,
		Args:       args,
	}
}

// ComputerOptions configures the anthropic.computer_* hosted tool family.
// Mirrors the ai-sdk argsSchemas at packages/anthropic/src/tool/computer_*.ts.
// EnableZoom is only honoured by computer_20251124.
type ComputerOptions struct {
	DisplayWidthPx  int   `json:"displayWidthPx"`
	DisplayHeightPx int   `json:"displayHeightPx"`
	DisplayNumber   *int  `json:"displayNumber,omitempty"`
	EnableZoom      *bool `json:"enableZoom,omitempty"`
}

// Computer20241022With returns a provider-defined computer_20241022 tool
// (requires computer-use-2024-10-22 beta).
func Computer20241022With(opts ComputerOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDComputer20241022,
		Args:       args,
	}
}

// Computer20250124With returns a provider-defined computer_20250124 tool
// (requires computer-use-2025-01-24 beta).
func Computer20250124With(opts ComputerOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDComputer20250124,
		Args:       args,
	}
}

// Computer20251124With returns a provider-defined computer_20251124 tool
// (requires computer-use-2025-11-24 beta). Set EnableZoom on opts to let
// the model issue the `zoom` action.
func Computer20251124With(opts ComputerOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDComputer20251124,
		Args:       args,
	}
}

// WebSearch20250305 returns a provider-defined web_search_20250305 tool.
// This predates the code-execution-web-tools beta and requires no header.
// Use WebSearch20250305With to pass options; the option shape is shared
// with WebSearchWith (see ai-sdk tool/web-search_20250305.ts).
func WebSearch20250305() tool.Tool { return WebSearch20250305With(WebSearchOptions{}) }

// WebSearch20250305With returns the tool with explicit options.
func WebSearch20250305With(opts WebSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDWebSearch20250305,
		Args:       args,
	}
}

// AdvisorOptions configures the anthropic.advisor_20260301 hosted tool.
// Mirrors the ai-sdk argsSchema at
// packages/anthropic/src/tool/advisor_20260301.ts. A faster, lower-cost
// executor model consults a higher-intelligence advisor model mid-generation
// for strategic guidance.
type AdvisorOptions struct {
	// Model is the advisor model ID (e.g. "claude-opus-4-7"). Required.
	// The advisor must be at least as capable as the executor.
	Model string `json:"model"`

	// MaxUses caps the number of advisor calls in a single request.
	MaxUses int `json:"maxUses,omitempty"`

	// Caching configures ephemeral caching of the advisor's view.
	Caching *AdvisorCaching `json:"caching,omitempty"`
}

// AdvisorCaching configures ephemeral caching for the advisor tool.
type AdvisorCaching struct {
	Type string `json:"type"` // "ephemeral"
	TTL  string `json:"ttl"`  // "5m" | "1h"
}

// Advisor returns a provider-defined Anthropic advisor_20260301 tool
// (requires the advisor-tool-2026-03-01 beta on the direct Anthropic API).
func Advisor(opts AdvisorOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDAdvisor20260301,
		Args:       args,
	}
}

// Wire-format shapes for each hosted tool. Each type field is the
// Anthropic API identifier; the `name` is the surface name the model
// will invoke.

type anthropicHostedCodeExecution20260120 struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedCodeExecution20250825 struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedWebSearch20260209 struct {
	Type           string               `json:"type"`
	Name           string               `json:"name"`
	MaxUses        int                  `json:"max_uses,omitempty"`
	AllowedDomains []string             `json:"allowed_domains,omitempty"`
	BlockedDomains []string             `json:"blocked_domains,omitempty"`
	UserLocation   *WebSearchUserLocale `json:"user_location,omitempty"`
}

type anthropicHostedWebFetch20260209 struct {
	Type             string   `json:"type"`
	Name             string   `json:"name"`
	MaxUses          int      `json:"max_uses,omitempty"`
	AllowedDomains   []string `json:"allowed_domains,omitempty"`
	BlockedDomains   []string `json:"blocked_domains,omitempty"`
	Citations        any      `json:"citations,omitempty"`
	MaxContentTokens int      `json:"max_content_tokens,omitempty"`
}

type anthropicHostedToolSearchRegex20251119 struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedToolSearchBM25_20251119 struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedBash struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedTextEditor struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicHostedTextEditor20250728 struct {
	Type          string `json:"type"`
	Name          string `json:"name"`
	MaxCharacters int    `json:"max_characters,omitempty"`
}

type anthropicHostedComputer struct {
	Type            string `json:"type"`
	Name            string `json:"name"`
	DisplayWidthPx  int    `json:"display_width_px"`
	DisplayHeightPx int    `json:"display_height_px"`
	DisplayNumber   *int   `json:"display_number,omitempty"`
}

type anthropicHostedComputer20251124 struct {
	Type            string `json:"type"`
	Name            string `json:"name"`
	DisplayWidthPx  int    `json:"display_width_px"`
	DisplayHeightPx int    `json:"display_height_px"`
	DisplayNumber   *int   `json:"display_number,omitempty"`
	EnableZoom      *bool  `json:"enable_zoom,omitempty"`
}

type anthropicHostedAdvisor20260301 struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`
	Model   string          `json:"model"`
	MaxUses int             `json:"max_uses,omitempty"`
	Caching *AdvisorCaching `json:"caching,omitempty"`
}

type anthropicHostedWebSearch20250305 struct {
	Type           string               `json:"type"`
	Name           string               `json:"name"`
	MaxUses        int                  `json:"max_uses,omitempty"`
	AllowedDomains []string             `json:"allowed_domains,omitempty"`
	BlockedDomains []string             `json:"blocked_domains,omitempty"`
	UserLocation   *WebSearchUserLocale `json:"user_location,omitempty"`
}

// ConvertProviderTool maps a goai provider-defined tool to its Anthropic
// wire-format payload and returns the beta header required by the tool
// on the direct Anthropic API (empty string if none). The third return
// is false when the tool's ProviderID isn't a known Anthropic hosted
// tool. Exported so Bedrock (and future Vertex Anthropic) can reuse the
// mapping.
func ConvertProviderTool(t tool.Tool) (any, string, bool) {
	switch t.ProviderID {
	case ToolIDCodeExecution20260120:
		return anthropicHostedCodeExecution20260120{
			Type: "code_execution_20260120",
			Name: "code_execution",
		}, "", true
	case ToolIDCodeExecution20250825:
		return anthropicHostedCodeExecution20250825{
			Type: "code_execution_20250825",
			Name: "code_execution",
		}, "code-execution-2025-08-25", true
	case ToolIDWebSearch20260209:
		var args WebSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedWebSearch20260209{
			Type:           "web_search_20260209",
			Name:           "web_search",
			MaxUses:        args.MaxUses,
			AllowedDomains: args.AllowedDomains,
			BlockedDomains: args.BlockedDomains,
			UserLocation:   args.UserLocation,
		}, "code-execution-web-tools-2026-02-09", true
	case ToolIDWebFetch20260209:
		var args WebFetchOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedWebFetch20260209{
			Type:             "web_fetch_20260209",
			Name:             "web_fetch",
			MaxUses:          args.MaxUses,
			AllowedDomains:   args.AllowedDomains,
			BlockedDomains:   args.BlockedDomains,
			Citations:        args.Citations,
			MaxContentTokens: args.MaxContentTokens,
		}, "code-execution-web-tools-2026-02-09", true
	case ToolIDToolSearchRegex20251119:
		return anthropicHostedToolSearchRegex20251119{
			Type: "tool_search_tool_regex_20251119",
			Name: "tool_search_tool_regex",
		}, "", true
	case ToolIDToolSearchBM25_20251119:
		return anthropicHostedToolSearchBM25_20251119{
			Type: "tool_search_tool_bm25_20251119",
			Name: "tool_search_tool_bm25",
		}, "", true
	case ToolIDBash20241022:
		return anthropicHostedBash{
			Type: "bash_20241022",
			Name: "bash",
		}, "computer-use-2024-10-22", true
	case ToolIDBash20250124:
		return anthropicHostedBash{
			Type: "bash_20250124",
			Name: "bash",
		}, "computer-use-2025-01-24", true
	case ToolIDTextEditor20241022:
		return anthropicHostedTextEditor{
			Type: "text_editor_20241022",
			Name: "str_replace_editor",
		}, "computer-use-2024-10-22", true
	case ToolIDTextEditor20250124:
		return anthropicHostedTextEditor{
			Type: "text_editor_20250124",
			Name: "str_replace_editor",
		}, "computer-use-2025-01-24", true
	case ToolIDTextEditor20250429:
		return anthropicHostedTextEditor{
			Type: "text_editor_20250429",
			Name: "str_replace_based_edit_tool",
		}, "computer-use-2025-01-24", true
	case ToolIDTextEditor20250728:
		var args TextEditorOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedTextEditor20250728{
			Type:          "text_editor_20250728",
			Name:          "str_replace_based_edit_tool",
			MaxCharacters: args.MaxCharacters,
		}, "", true
	case ToolIDComputer20241022:
		var args ComputerOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedComputer{
			Type:            "computer_20241022",
			Name:            "computer",
			DisplayWidthPx:  args.DisplayWidthPx,
			DisplayHeightPx: args.DisplayHeightPx,
			DisplayNumber:   args.DisplayNumber,
		}, "computer-use-2024-10-22", true
	case ToolIDComputer20250124:
		var args ComputerOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedComputer{
			Type:            "computer_20250124",
			Name:            "computer",
			DisplayWidthPx:  args.DisplayWidthPx,
			DisplayHeightPx: args.DisplayHeightPx,
			DisplayNumber:   args.DisplayNumber,
		}, "computer-use-2025-01-24", true
	case ToolIDComputer20251124:
		var args ComputerOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedComputer20251124{
			Type:            "computer_20251124",
			Name:            "computer",
			DisplayWidthPx:  args.DisplayWidthPx,
			DisplayHeightPx: args.DisplayHeightPx,
			DisplayNumber:   args.DisplayNumber,
			EnableZoom:      args.EnableZoom,
		}, "computer-use-2025-11-24", true
	case ToolIDAdvisor20260301:
		var args AdvisorOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedAdvisor20260301{
			Type:    "advisor_20260301",
			Name:    "advisor",
			Model:   args.Model,
			MaxUses: args.MaxUses,
			Caching: args.Caching,
		}, "advisor-tool-2026-03-01", true
	case ToolIDWebSearch20250305:
		var args WebSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return anthropicHostedWebSearch20250305{
			Type:           "web_search_20250305",
			Name:           "web_search",
			MaxUses:        args.MaxUses,
			AllowedDomains: args.AllowedDomains,
			BlockedDomains: args.BlockedDomains,
			UserLocation:   args.UserLocation,
		}, "", true
	}
	return nil, "", false
}
