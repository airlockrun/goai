package openai

import (
	"encoding/json"

	"github.com/airlockrun/goai/tool"
)

// Provider-defined tool IDs for OpenAI hosted tools. Mirror ai-sdk's
// factory IDs at packages/openai/src/tool/*.ts.
const (
	ToolIDWebSearch  = "openai.web_search"
	ToolIDCustom     = "openai.custom"
	ToolIDToolSearch = "openai.tool_search"
)

// WebSearchOptions configures the openai.web_search hosted tool.
// Mirrors the ai-sdk argsSchema at packages/openai/src/tool/web-search.ts.
type WebSearchOptions struct {
	ExternalWebAccess *bool                  `json:"externalWebAccess,omitempty"`
	Filters           *WebSearchFilters      `json:"filters,omitempty"`
	SearchContextSize string                 `json:"searchContextSize,omitempty"`
	UserLocation      *WebSearchUserLocation `json:"userLocation,omitempty"`
}

// WebSearchFilters narrows search results to specific domains.
type WebSearchFilters struct {
	AllowedDomains []string `json:"allowedDomains,omitempty"`
}

// WebSearchUserLocation provides geographically relevant search results.
// Type is always "approximate".
type WebSearchUserLocation struct {
	Type     string `json:"type"`
	Country  string `json:"country,omitempty"`
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearch returns a provider-defined OpenAI web_search tool. Use
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

// CustomOptions configures the openai.custom hosted tool.
// Mirrors the ai-sdk argsSchema at packages/openai/src/tool/custom.ts.
type CustomOptions struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Format      *CustomFormat `json:"format,omitempty"`
}

// CustomFormat specifies the output format for a custom tool. Either a
// grammar format (regex/lark) or a plain text format.
type CustomFormat struct {
	Type       string `json:"type"`
	Syntax     string `json:"syntax,omitempty"`
	Definition string `json:"definition,omitempty"`
}

// Custom returns a provider-defined OpenAI custom tool. The surface
// Name used by goai may differ from opts.Name (the wire-format name
// OpenAI knows) to support alias mapping.
func Custom(opts CustomOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDCustom,
		Name:       opts.Name,
		Args:       args,
	}
}

// ToolSearchOptions configures the openai.tool_search hosted tool.
// Mirrors the ai-sdk argsSchema at packages/openai/src/tool/tool-search.ts.
type ToolSearchOptions struct {
	Execution   string         `json:"execution,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolSearch returns a provider-defined OpenAI tool_search tool.
func ToolSearch(opts ToolSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDToolSearch,
		Args:       args,
	}
}

// Wire-format shapes for each hosted tool. Each type field is the
// OpenAI Responses API identifier.

type openaiHostedWebSearch struct {
	Type              string                     `json:"type"`
	ExternalWebAccess *bool                      `json:"external_web_access,omitempty"`
	Filters           *openaiHostedWebSearchFltr `json:"filters,omitempty"`
	SearchContextSize string                     `json:"search_context_size,omitempty"`
	UserLocation      *WebSearchUserLocation     `json:"user_location,omitempty"`
}

type openaiHostedWebSearchFltr struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

type openaiHostedCustom struct {
	Type        string        `json:"type"`
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Format      *CustomFormat `json:"format,omitempty"`
}

type openaiHostedToolSearch struct {
	Type        string         `json:"type"`
	Execution   string         `json:"execution,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// convertOpenAIProviderTool maps a goai provider-defined tool to its
// OpenAI Responses wire-format payload. The second return is false when
// the tool's ProviderID isn't a known OpenAI hosted tool; callers
// silently skip unknown provider tools.
func convertOpenAIProviderTool(t tool.Tool) (responsesToolWire, bool) {
	switch t.ProviderID {
	case ToolIDWebSearch:
		var args WebSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		var filters *openaiHostedWebSearchFltr
		if args.Filters != nil {
			filters = &openaiHostedWebSearchFltr{
				AllowedDomains: args.Filters.AllowedDomains,
			}
		}
		return openaiHostedWebSearch{
			Type:              "web_search",
			ExternalWebAccess: args.ExternalWebAccess,
			Filters:           filters,
			SearchContextSize: args.SearchContextSize,
			UserLocation:      args.UserLocation,
		}, true
	case ToolIDCustom:
		var args CustomOptions
		_ = json.Unmarshal(t.Args, &args)
		return openaiHostedCustom{
			Type:        "custom",
			Name:        args.Name,
			Description: args.Description,
			Format:      args.Format,
		}, true
	case ToolIDToolSearch:
		var args ToolSearchOptions
		_ = json.Unmarshal(t.Args, &args)
		return openaiHostedToolSearch{
			Type:        "tool_search",
			Execution:   args.Execution,
			Description: args.Description,
			Parameters:  args.Parameters,
		}, true
	}
	return nil, false
}
