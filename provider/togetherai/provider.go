// Package togetherai provides a Together AI provider implementation.
// Together AI uses an OpenAI-compatible API.
package togetherai

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.together.xyz/v1"
)

// Options contains configuration for the Together AI provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Together AI provider.
type Provider struct {
	compat *openaicompat.Provider
	opts   Options
}

// New creates a new Together AI provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		opts: opts,
		compat: openaicompat.New(openaicompat.Options{
			ProviderID: "togetherai",
			BaseURL:    baseURL,
			APIKey:     opts.APIKey,
			Headers:    opts.Headers,
		}),
	}
}

func (p *Provider) ID() string                                                 { return "togetherai" }
func (p *Provider) Model(modelID string) stream.Model                          { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel           { return p.Model(modelID) }
func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)
