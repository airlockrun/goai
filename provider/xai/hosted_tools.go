package xai

import (
	"encoding/json"

	"github.com/airlockrun/goai/tool"
)

// Provider-defined tool IDs for xAI hosted tools. Mirror ai-sdk's
// factory IDs at packages/xai/src/tool/*.ts.
const (
	ToolIDWebSearch     = "xai.web_search"
	ToolIDXSearch       = "xai.x_search"
	ToolIDCodeExecution = "xai.code_execution"
	ToolIDViewImage     = "xai.view_image"
	ToolIDViewXVideo    = "xai.view_x_video"
	ToolIDFileSearch    = "xai.file_search"
	ToolIDMCP           = "xai.mcp"
)

// WebSearchOptions configures the xai.web_search hosted tool.
// Mirrors the ai-sdk argsSchema at packages/xai/src/tool/web-search.ts.
type WebSearchOptions struct {
	AllowedDomains           []string `json:"allowedDomains,omitempty"`
	ExcludedDomains          []string `json:"excludedDomains,omitempty"`
	EnableImageUnderstanding *bool    `json:"enableImageUnderstanding,omitempty"`
}

// WebSearch returns a provider-defined xAI web_search tool. Use
// WebSearchWith for options.
func WebSearch() tool.Tool { return WebSearchWith(WebSearchOptions{}) }

func WebSearchWith(opts WebSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDWebSearch,
		Args:       args,
	}
}

// XSearchOptions configures the xai.x_search hosted tool.
// Mirrors the ai-sdk argsSchema at packages/xai/src/tool/x-search.ts.
type XSearchOptions struct {
	AllowedXHandles          []string `json:"allowedXHandles,omitempty"`
	ExcludedXHandles         []string `json:"excludedXHandles,omitempty"`
	FromDate                 string   `json:"fromDate,omitempty"`
	ToDate                   string   `json:"toDate,omitempty"`
	EnableImageUnderstanding *bool    `json:"enableImageUnderstanding,omitempty"`
	EnableVideoUnderstanding *bool    `json:"enableVideoUnderstanding,omitempty"`
}

// XSearch returns a provider-defined xAI x_search tool. Use XSearchWith
// for options.
func XSearch() tool.Tool { return XSearchWith(XSearchOptions{}) }

func XSearchWith(opts XSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDXSearch,
		Args:       args,
	}
}

// CodeExecution returns a provider-defined xAI code_execution tool.
// Wire-format type is "code_interpreter".
func CodeExecution() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDCodeExecution,
		Args:       json.RawMessage("{}"),
	}
}

// ViewImage returns a provider-defined xAI view_image tool.
func ViewImage() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDViewImage,
		Args:       json.RawMessage("{}"),
	}
}

// ViewXVideo returns a provider-defined xAI view_x_video tool.
func ViewXVideo() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDViewXVideo,
		Args:       json.RawMessage("{}"),
	}
}

// FileSearchOptions configures the xai.file_search hosted tool.
// Mirrors the ai-sdk argsSchema at packages/xai/src/tool/file-search.ts.
type FileSearchOptions struct {
	VectorStoreIDs []string `json:"vectorStoreIds"`
	MaxNumResults  *int     `json:"maxNumResults,omitempty"`
}

// FileSearch returns a provider-defined xAI file_search tool.
func FileSearch(opts FileSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDFileSearch,
		Args:       args,
	}
}

// MCPOptions configures the xai.mcp hosted tool.
// Mirrors the ai-sdk argsSchema at packages/xai/src/tool/mcp-server.ts.
type MCPOptions struct {
	ServerURL         string            `json:"serverUrl"`
	ServerLabel       string            `json:"serverLabel,omitempty"`
	ServerDescription string            `json:"serverDescription,omitempty"`
	AllowedTools      []string          `json:"allowedTools,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
}

// MCP returns a provider-defined xAI mcp tool.
func MCP(opts MCPOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDMCP,
		Args:       args,
	}
}

// Wire-format shapes for each hosted tool. Each type field is the xAI
// Responses API identifier.

type xaiHostedWebSearch struct {
	Type                     string   `json:"type"`
	AllowedDomains           []string `json:"allowed_domains,omitempty"`
	ExcludedDomains          []string `json:"excluded_domains,omitempty"`
	EnableImageUnderstanding *bool    `json:"enable_image_understanding,omitempty"`
}

type xaiHostedXSearch struct {
	Type                     string   `json:"type"`
	AllowedXHandles          []string `json:"allowed_x_handles,omitempty"`
	ExcludedXHandles         []string `json:"excluded_x_handles,omitempty"`
	FromDate                 string   `json:"from_date,omitempty"`
	ToDate                   string   `json:"to_date,omitempty"`
	EnableImageUnderstanding *bool    `json:"enable_image_understanding,omitempty"`
	EnableVideoUnderstanding *bool    `json:"enable_video_understanding,omitempty"`
}

type xaiHostedCodeInterpreter struct {
	Type string `json:"type"`
}

type xaiHostedViewImage struct {
	Type string `json:"type"`
}

type xaiHostedViewXVideo struct {
	Type string `json:"type"`
}

type xaiHostedFileSearch struct {
	Type           string   `json:"type"`
	VectorStoreIDs []string `json:"vector_store_ids,omitempty"`
	MaxNumResults  *int     `json:"max_num_results,omitempty"`
}

type xaiHostedMCP struct {
	Type              string            `json:"type"`
	ServerURL         string            `json:"server_url"`
	ServerLabel       string            `json:"server_label,omitempty"`
	ServerDescription string            `json:"server_description,omitempty"`
	AllowedTools      []string          `json:"allowed_tools,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
}

// convertXaiProviderTool maps a goai provider-defined tool to its xAI
// Responses wire-format payload. The second return is false when the
// tool's ProviderID isn't a known xAI hosted tool; callers silently
// skip unknown provider tools.
func convertXaiProviderTool(t tool.Tool) (responsesToolWire, bool) {
	switch t.ProviderID {
	case ToolIDWebSearch:
		var args WebSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return xaiHostedWebSearch{
			Type:                     "web_search",
			AllowedDomains:           args.AllowedDomains,
			ExcludedDomains:          args.ExcludedDomains,
			EnableImageUnderstanding: args.EnableImageUnderstanding,
		}, true
	case ToolIDXSearch:
		var args XSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return xaiHostedXSearch{
			Type:                     "x_search",
			AllowedXHandles:          args.AllowedXHandles,
			ExcludedXHandles:         args.ExcludedXHandles,
			FromDate:                 args.FromDate,
			ToDate:                   args.ToDate,
			EnableImageUnderstanding: args.EnableImageUnderstanding,
			EnableVideoUnderstanding: args.EnableVideoUnderstanding,
		}, true
	case ToolIDCodeExecution:
		return xaiHostedCodeInterpreter{Type: "code_interpreter"}, true
	case ToolIDViewImage:
		return xaiHostedViewImage{Type: "view_image"}, true
	case ToolIDViewXVideo:
		return xaiHostedViewXVideo{Type: "view_x_video"}, true
	case ToolIDFileSearch:
		var args FileSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return xaiHostedFileSearch{
			Type:           "file_search",
			VectorStoreIDs: args.VectorStoreIDs,
			MaxNumResults:  args.MaxNumResults,
		}, true
	case ToolIDMCP:
		var args MCPOptions
		_ = json.Unmarshal(t.Args, &args)
		return xaiHostedMCP{
			Type:              "mcp",
			ServerURL:         args.ServerURL,
			ServerLabel:       args.ServerLabel,
			ServerDescription: args.ServerDescription,
			AllowedTools:      args.AllowedTools,
			Headers:           args.Headers,
			Authorization:     args.Authorization,
		}, true
	}
	return nil, false
}

// isXaiHostedToolID reports whether the provider-defined tool ID names
// a server-side xAI tool. Used to short-circuit toolChoice forcing (the
// xAI Responses API does not support forcing hosted tools via
// tool_choice — only function tools can be forced).
func isXaiHostedToolID(id string) bool {
	switch id {
	case ToolIDWebSearch,
		ToolIDXSearch,
		ToolIDCodeExecution,
		ToolIDViewImage,
		ToolIDViewXVideo,
		ToolIDFileSearch,
		ToolIDMCP:
		return true
	}
	return false
}
