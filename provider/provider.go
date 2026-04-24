// Package provider defines the interface for LLM providers.
package provider

import (
	"context"
	"encoding/json"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Provider is the interface that all LLM providers must implement.
// At minimum, providers must implement LanguageModel support.
// Other model types are optional and return nil if not supported.
type Provider interface {
	// ID returns the provider identifier (e.g., "openai", "anthropic").
	ID() string

	// Model returns a language model instance for the given model ID.
	// This is the primary method for getting a model.
	Model(modelID string) stream.Model

	// LanguageModel returns a language model instance.
	// Returns nil if the model is not supported.
	LanguageModel(modelID string) model.LanguageModel

	// ImageModel returns an image generation model instance.
	// Returns nil if the provider doesn't support image generation.
	ImageModel(modelID string) model.ImageModel

	// EmbeddingModel returns an embedding model instance.
	// Returns nil if the provider doesn't support embeddings.
	EmbeddingModel(modelID string) model.EmbeddingModel

	// SpeechModel returns a text-to-speech model instance.
	// Returns nil if the provider doesn't support speech generation.
	SpeechModel(modelID string) model.SpeechModel

	// TranscriptionModel returns a speech-to-text model instance.
	// Returns nil if the provider doesn't support transcription.
	TranscriptionModel(modelID string) model.TranscriptionModel

	// RerankingModel returns a reranking model instance.
	// Returns nil if the provider doesn't support reranking.
	RerankingModel(modelID string) model.RerankingModel

	// Models returns a list of available model IDs.
	Models() []string
}

// LanguageProvider is a provider that supports language models.
// All providers must implement this.
type LanguageProvider interface {
	LanguageModel(modelID string) model.LanguageModel
}

// ImageProvider is a provider that supports image generation.
type ImageProvider interface {
	ImageModel(modelID string) model.ImageModel
}

// EmbeddingProvider is a provider that supports embeddings.
type EmbeddingProvider interface {
	EmbeddingModel(modelID string) model.EmbeddingModel
}

// SpeechProvider is a provider that supports text-to-speech.
type SpeechProvider interface {
	SpeechModel(modelID string) model.SpeechModel
}

// TranscriptionProvider is a provider that supports speech-to-text.
type TranscriptionProvider interface {
	TranscriptionModel(modelID string) model.TranscriptionModel
}

// RerankingProvider is a provider that supports document reranking.
type RerankingProvider interface {
	RerankingModel(modelID string) model.RerankingModel
}

// Options contains common provider configuration.
type Options struct {
	// APIKey is the API key for authentication.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string

	// Organization is the organization ID (for providers that support it).
	Organization string

	// Project is the project ID (for providers that support it, e.g., OpenAI).
	Project string
}

// ModelInfo contains metadata about a model.
type ModelInfo struct {
	ID           string
	Name         string
	Provider     string
	ContextLimit int
	OutputLimit  int
	Capabilities Capabilities
}

// Capabilities describes what a model supports.
type Capabilities struct {
	Tools       bool // Supports tool/function calling
	Vision      bool // Supports image input
	Streaming   bool // Supports streaming responses
	Reasoning   bool // Supports extended thinking/reasoning
	Temperature bool // Supports temperature parameter
}

// Registry holds registered providers.
type Registry struct {
	providers map[string]Provider
}

// NewRegistry creates a new provider registry.
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

// Register adds a provider to the registry.
func (r *Registry) Register(p Provider) {
	r.providers[p.ID()] = p
}

// Get returns a provider by ID.
func (r *Registry) Get(id string) (Provider, bool) {
	p, ok := r.providers[id]
	return p, ok
}

// GetModel returns a model from a provider.
// The modelID format is "provider/model" (e.g., "openai/gpt-4o").
func (r *Registry) GetModel(providerID, modelID string) (stream.Model, bool) {
	p, ok := r.providers[providerID]
	if !ok {
		return nil, false
	}
	return p.Model(modelID), true
}

// StreamInput wraps the input for provider streaming.
type StreamInput struct {
	*stream.Input
	Context context.Context
}

// ParseProviderOptions parses a map[string]any into a typed options struct.
// This is a Go-idiomatic way to handle provider-specific options.
// Usage:
//
//	opts, err := provider.ParseProviderOptions[ResponsesOptions](options.ProviderOptions)
//	if err != nil {
//	    return err
//	}
//	if opts.ReasoningEffort != "" {
//	    req.Reasoning = &reasoningConfig{Effort: opts.ReasoningEffort}
//	}
func ParseProviderOptions[T any](m map[string]any) (*T, error) {
	if m == nil {
		return new(T), nil
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	var opts T
	if err := json.Unmarshal(data, &opts); err != nil {
		return nil, err
	}
	return &opts, nil
}
