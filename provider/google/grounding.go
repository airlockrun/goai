package google

import (
	"encoding/json"

	"github.com/airlockrun/goai/tool"
)

// Provider-defined tool IDs for Google's built-in grounding tools. Mirrors
// ai-sdk's createProviderToolFactory ids in packages/google/src/tool/.
const (
	ToolIDGoogleSearch        = "google.google_search"
	ToolIDGoogleMaps          = "google.google_maps"
	ToolIDEnterpriseWebSearch = "google.enterprise_web_search"
	ToolIDURLContext          = "google.url_context"
	ToolIDCodeExecution       = "google.code_execution"
)

// GoogleSearchOptions configures the google_search grounding tool.
//
// Mode and DynamicThreshold are only honored on older Gemini versions
// that use the googleSearchRetrieval API (gemini-1.5-flash). Gemini 2+
// ignores them — the request becomes {googleSearch:{}} (possibly with
// searchTypes / timeRangeFilter) regardless.
//
// SearchTypes and TimeRangeFilter mirror ai-sdk #2565e70, which added
// explicit sub-selectors on the Gemini 2+ payload (letting callers
// opt into image search, web search, or narrow the time window).
type GoogleSearchOptions struct {
	// Mode is "MODE_DYNAMIC" or "MODE_UNSPECIFIED".
	Mode string `json:"mode,omitempty"`
	// DynamicThreshold is the similarity threshold for dynamic retrieval.
	DynamicThreshold float64 `json:"dynamicThreshold,omitempty"`

	// SearchTypes opts into specific grounding sub-queries (web, image).
	// Gemini 2+ only.
	SearchTypes *GoogleSearchTypes `json:"searchTypes,omitempty"`

	// TimeRangeFilter narrows results to a window. Both start and end
	// are RFC3339 timestamps.
	TimeRangeFilter *GoogleSearchTimeRange `json:"timeRangeFilter,omitempty"`
}

// GoogleSearchTypes selects which Google Search sub-queries to run.
// Fields are marker objects — presence enables the type.
type GoogleSearchTypes struct {
	WebSearch   *struct{} `json:"webSearch,omitempty"`
	ImageSearch *struct{} `json:"imageSearch,omitempty"`
}

// GoogleSearchTimeRange narrows grounding results to a date range.
type GoogleSearchTimeRange struct {
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}

// GoogleSearch returns the google_search grounding tool with no options.
// Suitable for Gemini 2+ where {googleSearch:{}} is the canonical form.
func GoogleSearch() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDGoogleSearch,
		Name:       "google_search",
	}
}

// GoogleSearchWith returns the google_search grounding tool with explicit
// options. The options are only sent on older Gemini models that still
// support googleSearchRetrieval; Gemini 2+ silently ignores them.
func GoogleSearchWith(opts GoogleSearchOptions) tool.Tool {
	args, _ := json.Marshal(opts)
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDGoogleSearch,
		Name:       "google_search",
		Args:       args,
	}
}

// GoogleMaps returns the google_maps grounding tool. Requires Gemini 2+.
func GoogleMaps() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDGoogleMaps,
		Name:       "google_maps",
	}
}

// EnterpriseWebSearch returns the enterprise_web_search grounding tool.
// Requires Gemini 2+.
func EnterpriseWebSearch() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDEnterpriseWebSearch,
		Name:       "enterprise_web_search",
	}
}

// URLContext returns the url_context tool. Requires Gemini 2+.
func URLContext() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDURLContext,
		Name:       "url_context",
	}
}

// CodeExecution returns the code_execution tool. Requires Gemini 2+.
func CodeExecution() tool.Tool {
	return tool.Tool{
		Type:       "provider",
		ProviderID: ToolIDCodeExecution,
		Name:       "code_execution",
	}
}
