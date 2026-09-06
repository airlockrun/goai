// Package cerebras provides a Cerebras provider implementation.
package cerebras

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.cerebras.ai/v1"
)

// Options contains configuration for the Cerebras provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Cerebras provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new Cerebras provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:                "cerebras",
			APIKey:                    opts.APIKey,
			BaseURL:                   baseURL,
			Headers:                   opts.Headers,
			SupportsStructuredOutputs: true,
			RequestModifier:           cerebrasRequestModifier,
			MessageConverter:          convertMessages,
			TransformRequest: func(_ string, body map[string]any) error {
				if value, ok := body["max_tokens"]; ok {
					body["max_completion_tokens"] = value
					delete(body, "max_tokens")
				}
				return nil
			},
		}),
	}
}

func (p *Provider) ID() string { return "cerebras" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.compat.Model(modelID)
}

func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)
