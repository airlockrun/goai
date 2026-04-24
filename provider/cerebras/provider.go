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

// Models returns the Cerebras model catalog. Mirrors
// ai-sdk/packages/cerebras/src/cerebras-chat-options.ts CerebrasChatModelId.
func (p *Provider) Models() []string {
	return []string{
		"llama3.1-8b",
		"qwen-3-235b-a22b-instruct-2507",
		"qwen-3-235b-a22b-thinking-2507",
		"zai-glm-4.6",
		"zai-glm-4.7",
	}
}

var _ provider.Provider = (*Provider)(nil)
