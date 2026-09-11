// Package groq provides a Groq provider implementation.
// Groq uses an OpenAI-compatible API.
package groq

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

// groqCallWarner emits Groq chat unsupported-option warnings.
// Mirrors ai-sdk packages/groq/src/groq-chat-language-model.ts.
func groqCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	return warnings
}

const (
	defaultBaseURL = "https://api.groq.com/openai/v1"
)

// Options contains configuration for the Groq provider.
type Options struct {
	// APIKey is the Groq API key.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Groq provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new Groq provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:      "groq",
			BaseURL:         baseURL,
			APIKey:          opts.APIKey,
			Headers:         opts.Headers,
			RequestModifier: groqRequestModifier,
			ReasoningMapper: func(id string, options *stream.CallOptions) (map[string]any, []stream.Warning) {
				if effort, _ := options.ProviderOptions["reasoningEffort"].(string); effort != "" {
					return nil, nil
				}
				values := map[string]string{"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}
				if id == "qwen/qwen3.6-27b" {
					values["none"] = "none"
				}
				effort, warnings := provider.MapReasoning(options.Reasoning, values)
				if effort == "" {
					return nil, warnings
				}
				return map[string]any{"reasoning_effort": effort}, warnings
			},
			CallWarner:                groqCallWarner,
			SupportsStructuredOutputs: true,
		}),
	}
}

// groqRequestModifier applies Groq-specific options to the request.
func groqRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	extra := make(map[string]any)

	if opts.ReasoningFormat != "" {
		extra["reasoning_format"] = opts.ReasoningFormat
	}
	if opts.ReasoningEffort != "" {
		extra["reasoning_effort"] = opts.ReasoningEffort
	}
	if opts.ServiceTier != "" {
		extra["service_tier"] = opts.ServiceTier
	}
	if opts.User != "" {
		extra["user"] = opts.User
	}
	if opts.ParallelToolCalls != nil {
		extra["parallel_tool_calls"] = *opts.ParallelToolCalls
	}

	return extra, nil, nil
}

// ID returns "groq".
func (p *Provider) ID() string {
	return "groq"
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns nil as Groq doesn't support image generation.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return nil
}

// EmbeddingModel returns nil as Groq doesn't support embeddings.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return nil
}

// SpeechModel returns nil as Groq doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns a transcription model instance.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &GroqTranscriptionModel{
		id:       modelID,
		provider: p,
	}
}

// RerankingModel returns nil as Groq doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

// Ensure Provider implements the provider interface
var _ provider.Provider = (*Provider)(nil)
