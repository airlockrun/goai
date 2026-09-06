// Package perplexity provides a Perplexity provider implementation.
// Perplexity uses an OpenAI-compatible API.
package perplexity

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.perplexity.ai"
)

// Options contains configuration for the Perplexity provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Perplexity provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new Perplexity provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID: "perplexity",
			BaseURL:    baseURL,
			APIKey:     opts.APIKey,
			Headers:    opts.Headers,
			CallWarner: perplexityCallWarner,
		}),
	}
}

// perplexityCallWarner emits unsupported-option warnings for Perplexity.
// Mirrors ai-sdk packages/perplexity/src/perplexity-language-model.ts.
func perplexityCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	if len(options.StopSequences) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("stopSequences", ""))
	}
	if options.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}
	return warnings
}

func (p *Provider) ID() string                                       { return "perplexity" }
func (p *Provider) Model(modelID string) stream.Model                { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return p.Model(modelID) }
func (p *Provider) ImageModel(modelID string) model.ImageModel       { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &PerplexityEmbeddingModel{id: modelID, provider: p}
}
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)
