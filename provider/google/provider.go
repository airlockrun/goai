// Package google provides a Google AI (Gemini) provider implementation.
package google

import (
	"github.com/airlockrun/goai/model"
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

// SpeechModel returns nil as Google AI doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns nil as Google AI doesn't support transcription.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return nil
}

// RerankingModel returns nil as Google AI doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

// Models returns available model IDs.
// Mirrors ai-sdk's GoogleGenerativeAIModelId union in
// packages/google/src/google-generative-ai-options.ts.
func (p *Provider) Models() []string {
	return []string{
		// Gemini 3.x
		"gemini-3-pro-preview",
		"gemini-3-pro-image-preview",
		"gemini-3-flash-preview",
		"gemini-3.1-pro-preview",
		"gemini-3.1-pro-preview-customtools",
		"gemini-3.1-flash-image-preview",
		"gemini-3.1-flash-lite-preview",
		"gemini-3.1-flash-tts-preview",
		"gemini-3.5-flash",
		// Gemini 2.5
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-image",
		"gemini-2.5-flash-lite",
		"gemini-2.5-flash-preview-tts",
		"gemini-2.5-pro-preview-tts",
		"gemini-2.5-computer-use-preview-10-2025",
		"gemini-2.5-flash-native-audio-latest",
		"gemini-2.5-flash-native-audio-preview-09-2025",
		"gemini-2.5-flash-native-audio-preview-12-2025",
		// Gemini 2.0
		"gemini-2.0-flash",
		"gemini-2.0-flash-001",
		"gemini-2.0-flash-lite",
		"gemini-2.0-flash-lite-001",
		// Rolling aliases
		"gemini-pro-latest",
		"gemini-flash-latest",
		"gemini-flash-lite-latest",
		// Gemini 1.5 (still active)
		"gemini-1.5-pro",
		"gemini-1.5-flash",
		"gemini-1.5-flash-8b",
		// Specialty previews
		"deep-research-pro-preview-12-2025",
		"deep-research-max-preview-04-2026",
		"deep-research-preview-04-2026",
		"nano-banana-pro-preview",
		// Gemma 3 open-weight family via Gemini API
		"gemma-3-1b-it",
		"gemma-3-4b-it",
		"gemma-3-12b-it",
		"gemma-3-27b-it",
		"gemma-3n-e2b-it",
		"gemma-3n-e4b-it",
	}
}

// EmbeddingModels returns available embedding model IDs.
func (p *Provider) EmbeddingModels() []string {
	return []string{
		"text-embedding-004",
		"embedding-001",
		"gemini-embedding-2",
	}
}
