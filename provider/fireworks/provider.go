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
			ProviderID: "fireworks",
			BaseURL:    baseURL,
			APIKey:     opts.APIKey,
			Headers:    opts.Headers,
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

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

// Models returns the Fireworks chat-model catalog. Mirrors ai-sdk's
// FireworksChatModelId union in packages/fireworks/src/fireworks-chat-options.ts.
func (p *Provider) Models() []string {
	return []string{
		"accounts/fireworks/models/deepseek-v3",
		"accounts/fireworks/models/llama-v3p3-70b-instruct",
		"accounts/fireworks/models/llama-v3p2-3b-instruct",
		"accounts/fireworks/models/llama-v3p1-405b-instruct",
		"accounts/fireworks/models/llama-v3p1-8b-instruct",
		"accounts/fireworks/models/mixtral-8x7b-instruct",
		"accounts/fireworks/models/mixtral-8x22b-instruct",
		"accounts/fireworks/models/mixtral-8x7b-instruct-hf",
		"accounts/fireworks/models/qwen2p5-coder-32b-instruct",
		"accounts/fireworks/models/qwen2p5-72b-instruct",
		"accounts/fireworks/models/qwen-qwq-32b-preview",
		"accounts/fireworks/models/qwen2-vl-72b-instruct",
		"accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
		"accounts/fireworks/models/qwq-32b",
		"accounts/fireworks/models/yi-large",
		"accounts/fireworks/models/kimi-k2-instruct",
		"accounts/fireworks/models/kimi-k2-thinking",
		"accounts/fireworks/models/kimi-k2p5",
		"accounts/fireworks/models/minimax-m2",
	}
}

var _ provider.Provider = (*Provider)(nil)
