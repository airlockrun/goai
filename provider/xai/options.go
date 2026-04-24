package xai

// ChatOptions contains provider-specific options for the xAI (Grok) API.
// These options match ai-sdk's XaiProviderOptions schema.
// See: ai-sdk/packages/xai/src/xai-chat-options.ts
type ChatOptions struct {
	// ReasoningEffort controls the reasoning effort level.
	// Values: "low", "high"
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// ParallelFunctionCalling controls whether to enable parallel function calling.
	// When true, the model can call multiple functions in parallel.
	// When false, the model will call functions sequentially.
	// Default is true.
	ParallelFunctionCalling *bool `json:"parallel_function_calling,omitempty"`

	// SearchParameters configures search behavior.
	SearchParameters *SearchParameters `json:"searchParameters,omitempty"`

	// Logprobs requests per-token log probabilities alongside the
	// generated text (ai-sdk #2e00e03).
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs, when set, returns the top-N alternative token
	// probabilities for each position. Max 8 (ai-sdk #2e00e03).
	TopLogprobs *int `json:"topLogprobs,omitempty"`
}

// SearchParameters configures xAI search behavior.
type SearchParameters struct {
	// Mode is the search mode preference.
	// Values: "off" (disables search), "auto" (model decides, default), "on" (always enables)
	Mode string `json:"mode"`

	// ReturnCitations controls whether to return citations in the response.
	// Default is true.
	ReturnCitations *bool `json:"returnCitations,omitempty"`

	// FromDate is the start date for search data (ISO8601 format: YYYY-MM-DD).
	FromDate string `json:"fromDate,omitempty"`

	// ToDate is the end date for search data (ISO8601 format: YYYY-MM-DD).
	ToDate string `json:"toDate,omitempty"`

	// MaxSearchResults is the maximum number of search results to consider.
	// Default is 20. Range: 1-50.
	MaxSearchResults int `json:"maxSearchResults,omitempty"`

	// Sources are the data sources to search from.
	// Default is [{ type: 'web' }, { type: 'x' }] if not specified.
	Sources []SearchSource `json:"sources,omitempty"`
}

// ResponsesOptions contains provider-specific options for the xAI
// Responses API (/v1/responses). Mirrors ai-sdk's
// xaiLanguageModelResponsesOptions schema.
// See: ai-sdk/packages/xai/src/responses/xai-responses-options.ts
type ResponsesOptions struct {
	// ReasoningEffort controls the reasoning effort level.
	// Values: "low", "medium", "high".
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// Logprobs requests per-token log probabilities alongside the
	// generated text.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs, when set, returns the top-N alternative token
	// probabilities for each position. Max 8.
	TopLogprobs *int `json:"topLogprobs,omitempty"`

	// Store controls whether to persist the input + response on xAI's
	// servers for later retrieval. Defaults to true. Must be false for
	// teams with Zero Data Retention (ZDR) enabled.
	Store *bool `json:"store,omitempty"`

	// PreviousResponseID chains this call to a prior response so xAI can
	// re-load the persisted reasoning / tool-call context.
	PreviousResponseID string `json:"previousResponseId,omitempty"`

	// Include requests additional fields in the response payload.
	// Example: "reasoning.encrypted_content", "file_search_call.results".
	Include []string `json:"include,omitempty"`
}

// SearchSource represents a search data source.
type SearchSource struct {
	// Type is the source type: "web", "x", "news", or "rss".
	Type string `json:"type"`

	// Country is the 2-letter country code (for "web" and "news" types).
	Country string `json:"country,omitempty"`

	// ExcludedWebsites is a list of websites to exclude (max 5, for "web" and "news" types).
	ExcludedWebsites []string `json:"excludedWebsites,omitempty"`

	// AllowedWebsites is a list of websites to include (max 5, for "web" type only).
	AllowedWebsites []string `json:"allowedWebsites,omitempty"`

	// SafeSearch enables safe search (for "web" and "news" types).
	SafeSearch *bool `json:"safeSearch,omitempty"`

	// ExcludedXHandles is a list of X handles to exclude (for "x" type).
	ExcludedXHandles []string `json:"excludedXHandles,omitempty"`

	// IncludedXHandles is a list of X handles to include (for "x" type).
	IncludedXHandles []string `json:"includedXHandles,omitempty"`

	// PostFavoriteCount is the minimum favorite count for posts (for "x" type).
	PostFavoriteCount int `json:"postFavoriteCount,omitempty"`

	// PostViewCount is the minimum view count for posts (for "x" type).
	PostViewCount int `json:"postViewCount,omitempty"`

	// Links is a list of RSS feed URLs (max 1, for "rss" type).
	Links []string `json:"links,omitempty"`
}
