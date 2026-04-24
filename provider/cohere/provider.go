// Package cohere provides a Cohere provider implementation.
package cohere

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.cohere.com/v1"
)

// Options contains configuration for the Cohere provider.
type Options struct {
	// APIKey is the Cohere API key.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Cohere provider.
type Provider struct {
	opts Options
}

// New creates a new Cohere provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

// ID returns "cohere".
func (p *Provider) ID() string {
	return "cohere"
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return &CohereModel{
		id:       modelID,
		provider: p,
	}
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns nil as Cohere doesn't support image generation.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return nil
}

// EmbeddingModel returns an embedding model instance.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &CohereEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

// SpeechModel returns nil as Cohere doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns nil as Cohere doesn't support transcription.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return nil
}

// RerankingModel returns a reranking model instance.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return &CohereRerankingModel{
		id:       modelID,
		provider: p,
	}
}

// Models returns available model IDs. Mirrors ai-sdk's CohereChatModelId
// union in packages/cohere/src/cohere-chat-options.ts.
func (p *Provider) Models() []string {
	return []string{
		"command-a-03-2025",
		"command-a-reasoning-08-2025",
		"command-r7b-12-2024",
		"command-r-plus-04-2024",
		"command-r-plus",
		"command-r-08-2024",
		"command-r-03-2024",
		"command-r",
		"command",
		"command-light",
	}
}

// EmbeddingModels returns available embedding model IDs.
func (p *Provider) EmbeddingModels() []string {
	return []string{
		"embed-english-v3.0",
		"embed-multilingual-v3.0",
		"embed-english-light-v3.0",
		"embed-multilingual-light-v3.0",
	}
}

// RerankingModels returns available reranking model IDs.
func (p *Provider) RerankingModels() []string {
	return []string{
		"rerank-english-v3.0",
		"rerank-multilingual-v3.0",
		"rerank-english-v2.0",
		"rerank-multilingual-v2.0",
	}
}

var _ provider.Provider = (*Provider)(nil)
