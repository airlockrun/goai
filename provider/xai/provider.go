// Package xai provides an xAI (Grok) provider implementation.
// xAI uses an OpenAI-compatible API.
package xai

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.x.ai/v1"
)

// responsesModelCatalog lists the xAI model IDs that speak the
// Responses API (/v1/responses). Mirrors ai-sdk's XaiResponsesModelId
// union plus goai-preferred reasoning IDs. Model() routes these IDs to
// the Responses implementation; all other IDs fall through to the Chat
// Completions / openaicompat path.
var responsesModelCatalog = map[string]bool{
	"grok-4":                       true,
	"grok-4-0709":                  true,
	"grok-4-latest":                true,
	"grok-4-1-fast-reasoning":      true,
	"grok-4-1-fast-non-reasoning":  true,
	"grok-4-fast-reasoning":        true,
	"grok-4-fast-non-reasoning":    true,
	"grok-4.20-0309-reasoning":     true,
	"grok-4.20-0309-non-reasoning": true,
	"grok-4.20-multi-agent-0309":   true,
	"grok-code-fast-1":             true,
}

// isResponsesModel reports whether the given model ID should be routed
// to the Responses API rather than Chat Completions.
func isResponsesModel(id string) bool {
	return responsesModelCatalog[id]
}

// Options contains configuration for the xAI provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the xAI provider.
type Provider struct {
	compat  *openaicompat.Provider
	baseURL string
	apiKey  string
	headers map[string]string
}

// New creates a new xAI provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:      "xai",
			BaseURL:         baseURL,
			APIKey:          opts.APIKey,
			Headers:         opts.Headers,
			RequestModifier: xaiRequestModifier,
			CallWarner:      xaiChatCallWarner,
		}),
		baseURL: baseURL,
		apiKey:  opts.APIKey,
		headers: opts.Headers,
	}
}

// xaiChatCallWarner emits the unsupported-option warnings for xAI Chat
// Completions. Mirrors ai-sdk packages/xai/src/xai-chat-language-model.ts.
func xaiChatCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if options.PresencePenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", ""))
	}
	if len(options.StopSequences) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("stopSequences", ""))
	}
	return warnings
}

// xaiRequestModifier applies xAI-specific options to the request.
// It also emits unsupported-option warnings for xAI Chat Completions
// (ai-sdk: packages/xai/src/xai-chat-language-model.ts).
func xaiRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	// The openaicompat base can't see CallOptions fields like topK /
	// frequencyPenalty, so warn via providerOptions mirror keys. ai-sdk
	// flags these on xAI chat but the RequestModifier contract only gets
	// providerOptions, so emit nothing from here — those live as chat-path
	// warnings surfaced in chat.buildRequest instead (see ai-sdk parity).
	extra := make(map[string]any)

	if opts.ReasoningEffort != "" {
		extra["reasoning_effort"] = opts.ReasoningEffort
	}
	if opts.ParallelFunctionCalling != nil {
		extra["parallel_function_calling"] = *opts.ParallelFunctionCalling
	}
	if opts.Logprobs != nil {
		extra["logprobs"] = *opts.Logprobs
	}
	if opts.TopLogprobs != nil {
		extra["top_logprobs"] = *opts.TopLogprobs
	}
	if opts.SearchParameters != nil {
		searchParams := make(map[string]any)
		if opts.SearchParameters.Mode != "" {
			searchParams["mode"] = opts.SearchParameters.Mode
		}
		if opts.SearchParameters.ReturnCitations != nil {
			searchParams["return_citations"] = *opts.SearchParameters.ReturnCitations
		}
		if opts.SearchParameters.FromDate != "" {
			searchParams["from_date"] = opts.SearchParameters.FromDate
		}
		if opts.SearchParameters.ToDate != "" {
			searchParams["to_date"] = opts.SearchParameters.ToDate
		}
		if opts.SearchParameters.MaxSearchResults > 0 {
			searchParams["max_search_results"] = opts.SearchParameters.MaxSearchResults
		}
		if len(opts.SearchParameters.Sources) > 0 {
			searchParams["sources"] = opts.SearchParameters.Sources
		}
		if len(searchParams) > 0 {
			extra["search_parameters"] = searchParams
		}
	}

	return extra, nil, nil
}

func (p *Provider) ID() string { return "xai" }

// Model returns a language model for the given ID. Grok 4 reasoning
// models (and grok-code-fast-1) are routed to the Responses API; all
// other models continue to flow through the Chat Completions / openaicompat
// path for backward compatibility with Grok 3 callers.
func (p *Provider) Model(modelID string) stream.Model {
	if isResponsesModel(modelID) {
		return p.Responses(modelID)
	}
	return p.Chat(modelID)
}

// LanguageModel returns a language model interface for the given ID.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return p.Model(modelID) }

// Chat returns an openaicompat-backed Chat Completions model. Always
// available; Model() picks this path for non-Responses IDs.
func (p *Provider) Chat(modelID string) stream.Model { return p.compat.Model(modelID) }

// Responses returns an XaiResponsesModel wired to /v1/responses.
func (p *Provider) Responses(modelID string) stream.Model {
	return &XaiResponsesModel{
		id:       modelID,
		provider: p,
		baseURL:  p.baseURL,
		apiKey:   p.apiKey,
		headers:  p.headers,
	}
}

// ImageModel returns an XaiImageModel wired to /v1/images/{generations,edits}.
// Supported catalog: grok-imagine-image, grok-imagine-image-pro.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &XaiImageModel{id: modelID, provider: p}
}
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

// Models returns available model IDs. Mirrors ai-sdk's XaiChatModelId
// union in packages/xai/src/xai-chat-options.ts.
//
// Obsolete Grok 2 and beta variants (grok-2, grok-2-mini, grok-beta) are
// pruned per ai-sdk #55ccbe2 + #64a8fae. The `grok-4-1` bare ID is
// NOT listed — only -fast-reasoning / -fast-non-reasoning exist
// (ai-sdk #05f3f36 corrected this).
func (p *Provider) Models() []string {
	return []string{
		// Grok 4.20 (latest)
		"grok-4.20-0309-non-reasoning",
		"grok-4.20-0309-reasoning",
		"grok-4.20-multi-agent-0309",
		// Grok 4.1 family
		"grok-4-1-fast-reasoning",
		"grok-4-1-fast-non-reasoning",
		// Grok 4 family
		"grok-4",
		"grok-4-0709",
		"grok-4-latest",
		"grok-4-fast-reasoning",
		"grok-4-fast-non-reasoning",
		// Grok 3 family
		"grok-3",
		"grok-3-latest",
		"grok-3-mini",
		"grok-3-mini-latest",
		// Grok code fast
		"grok-code-fast-1",
	}
}

var _ provider.Provider = (*Provider)(nil)
