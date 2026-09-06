// Package google provides a Google AI (Gemini) provider implementation.
package google

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
)

// Options contains configuration for the Google provider.
type Options struct {
	// APIKey is the Google AI API key.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Google AI provider.
type Provider struct {
	opts Options
}

// New creates a new Google AI provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

// ID returns "google".
func (p *Provider) ID() string {
	return "google"
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return &GoogleModel{
		id:       modelID,
		provider: p,
	}
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns an image model instance.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &GoogleImageModel{
		id:       modelID,
		provider: p,
	}
}

// EmbeddingModel returns an embedding model instance.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &GoogleEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

// SpeechModel returns a Gemini text-to-speech model.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &GoogleSpeechModel{id: modelID, provider: p}
}

// TranscriptionModel returns a unary Gemini transcription model.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &GoogleTranscriptionModel{id: modelID, provider: p}
}

// RerankingModel returns nil as Google AI doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

var _ provider.Provider = (*Provider)(nil)
