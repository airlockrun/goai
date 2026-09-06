// Package fireworks provides a Fireworks AI provider implementation.
// Fireworks uses an OpenAI-compatible API.
package fireworks

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.fireworks.ai/inference/v1"
)

// Options contains configuration for the Fireworks provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string

	// PollIntervalMS overrides the async image polling interval. Defaults
	// to 500ms when zero. Mirrors ai-sdk's pollIntervalMillis.
	PollIntervalMS int

	// PollTimeoutMS overrides the async image polling timeout. Defaults
	// to 120000ms (2 minutes) when zero. Mirrors ai-sdk's pollTimeoutMillis.
	PollTimeoutMS int
}

// Provider implements the Fireworks provider.
type Provider struct {
	compat         *openaicompat.Provider
	baseURL        string
	apiKey         string
	headers        map[string]string
	pollIntervalMS int
	pollTimeoutMS  int
}

// New creates a new Fireworks provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:                "fireworks",
			BaseURL:                   baseURL,
			APIKey:                    opts.APIKey,
			Headers:                   opts.Headers,
			SupportsStructuredOutputs: true,
			RequestModifier:           fireworksRequestModifier,
			TransformRequest: func(_ string, body map[string]any) error {
				switch body["reasoning_effort"] {
				case "minimal":
					body["reasoning_effort"] = "low"
				case "xhigh":
					body["reasoning_effort"] = "high"
				}
				return nil
			},
		}),
		baseURL:        baseURL,
		apiKey:         opts.APIKey,
		headers:        opts.Headers,
		pollIntervalMS: opts.PollIntervalMS,
		pollTimeoutMS:  opts.PollTimeoutMS,
	}
}

func (p *Provider) ID() string                                       { return "fireworks" }
func (p *Provider) Model(modelID string) stream.Model                { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return p.Model(modelID) }

// ImageModel returns a FireworksImageModel wired to the workflows /
// image_generation endpoints. Mirrors ai-sdk's FireworksImageModel.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &FireworksImageModel{
		id:             modelID,
		provider:       p,
		pollIntervalMS: p.pollIntervalMS,
		pollTimeoutMS:  p.pollTimeoutMS,
	}
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return p.compat.EmbeddingModel(modelID)
}
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)
