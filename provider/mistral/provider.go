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
	strictJSONSchema := false
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
			DefaultStrictJSONSchema:   &strictJSONSchema,
			SupportsPenalties:         true,
			MessageConverter:          convertMessages,
			TransformRequest: func(id string, body map[string]any) error {
				if !supportsReasoningEffort(id) {
					delete(body, "reasoning_effort")
					return nil
				}
				if effort, ok := body["reasoning_effort"].(string); ok && effort != "none" {
					body["reasoning_effort"] = "high"
				}
				return nil
			},
			ReasoningMapper: func(id string, options *stream.CallOptions) (map[string]any, []stream.Warning) {
				if effort, _ := options.ProviderOptions["reasoningEffort"].(string); effort != "" {
					return nil, nil
				}
				values := map[string]string{"none": "none", "minimal": "high", "low": "high", "medium": "high", "high": "high", "xhigh": "high"}
				if !supportsReasoningEffort(id) {
					values = nil
				}
				effort, warnings := provider.MapReasoning(options.Reasoning, values)
				if effort == "" {
					return nil, warnings
				}
				return map[string]any{"reasoning_effort": effort}, warnings
			},
		}),
	}
}

func supportsReasoningEffort(id string) bool {
	return id == "mistral-small-latest" || id == "mistral-small-2603" || id == "mistral-medium-3" || id == "mistral-medium-3.5"
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

// SpeechModel returns a speech generation model.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &MistralSpeechModel{id: modelID, provider: p}
}

// TranscriptionModel returns an audio transcription model.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &MistralTranscriptionModel{id: modelID, provider: p}
}

// RerankingModel returns nil as Mistral doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

var _ provider.Provider = (*Provider)(nil)
