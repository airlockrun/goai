// Package mistral provides a Mistral AI provider implementation.
package mistral

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.mistral.ai/v1"
)

// Options contains configuration for the Mistral provider.
type Options struct {
	// APIKey is the Mistral API key.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Mistral provider.
type Provider struct {
	compat *openaicompat.Provider
	opts   Options
}

// New creates a new Mistral provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		opts: opts,
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:                "mistral",
			BaseURL:                   baseURL,
			APIKey:                    opts.APIKey,
			Headers:                   opts.Headers,
			RequestModifier:           mistralRequestModifier,
			CallWarner:                mistralCallWarner,
			SupportsStructuredOutputs: true,
		}),
	}
}

// mistralRequestModifier applies Mistral-specific options to the request.
func mistralRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	extra := make(map[string]any)

	if opts.SafePrompt != nil {
		extra["safe_prompt"] = *opts.SafePrompt
	}
	if opts.ParallelToolCalls != nil {
		extra["parallel_tool_calls"] = *opts.ParallelToolCalls
	}
	// reasoningEffort toggles reasoning on mistral-small-latest et al.
	// (ai-sdk #297e685). "high" enables, "none" disables.
	if opts.ReasoningEffort != "" {
		extra["reasoning_effort"] = opts.ReasoningEffort
	}

	return extra, nil, nil
}

// mistralCallWarner emits Mistral unsupported-option warnings.
// Mirrors ai-sdk packages/mistral/src/mistral-chat-language-model.ts.
func mistralCallWarner(options *stream.CallOptions) []stream.Warning {
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
	return warnings
}

// ID returns "mistral".
func (p *Provider) ID() string {
	return "mistral"
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns nil as Mistral doesn't support image generation.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return nil
}

// EmbeddingModel returns an embedding model instance.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &MistralEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

// SpeechModel returns nil as Mistral doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns nil as Mistral doesn't support transcription.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return nil
}

// RerankingModel returns nil as Mistral doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

var _ provider.Provider = (*Provider)(nil)
