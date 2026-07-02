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

var _ provider.Provider = (*Provider)(nil)
