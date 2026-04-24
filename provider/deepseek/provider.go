// Package deepseek provides a DeepSeek provider implementation.
// DeepSeek uses an OpenAI-compatible API.
package deepseek

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.deepseek.com/v1"
)

// Options contains configuration for the DeepSeek provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the DeepSeek provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new DeepSeek provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:      "deepseek",
			BaseURL:         baseURL,
			APIKey:          opts.APIKey,
			Headers:         opts.Headers,
			RequestModifier: deepseekRequestModifier,
			CallWarner:      deepseekCallWarner,
		}),
	}
}

// deepseekRequestModifier applies DeepSeek-specific options to the request.
func deepseekRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	extra := make(map[string]any)

	// DeepSeek uses "enable_thinking" boolean in the API
	if opts.Thinking != nil {
		if opts.Thinking.Type == "disabled" {
			extra["enable_thinking"] = false
		} else if opts.Thinking.Type == "enabled" {
			extra["enable_thinking"] = true
		}
	}

	return extra, nil, nil
}

// deepseekCallWarner emits DeepSeek chat unsupported-option warnings.
func deepseekCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	return warnings
}

func (p *Provider) ID() string                                                 { return "deepseek" }
func (p *Provider) Model(modelID string) stream.Model                          { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel           { return p.Model(modelID) }
func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{"deepseek-chat", "deepseek-coder", "deepseek-reasoner"}
}

var _ provider.Provider = (*Provider)(nil)
