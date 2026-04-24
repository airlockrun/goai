// Package anthropic provides an Anthropic provider implementation.
package anthropic

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.anthropic.com/v1"
	apiVersion     = "2023-06-01"
)

// Options contains configuration for the Anthropic provider.
type Options struct {
	// APIKey is the credential used for authentication. The default auth
	// scheme sends it as an x-api-key header (direct Anthropic). When
	// AuthScheme is "bearer", it is sent as Authorization: Bearer {APIKey}
	// instead (Vertex Anthropic).
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string

	// AuthScheme controls how APIKey is placed on outgoing requests. The
	// zero value ("" or "api-key") sends x-api-key, matching direct
	// Anthropic. "bearer" sends Authorization: Bearer {APIKey}, used by
	// derived providers like Vertex Anthropic.
	AuthScheme string
}

// Provider implements the Anthropic provider.
type Provider struct {
	opts Options
	cfg  Config
}

// New creates a new Anthropic provider for the direct Anthropic API. Derived
// providers (Bedrock Anthropic, Vertex Anthropic) should call NewWithConfig.
func New(opts Options) *Provider {
	return NewWithConfig(opts, Config{})
}

// NewWithConfig creates a new Anthropic provider with explicit hook
// configuration. The zero-value Config targets the direct Anthropic API.
func NewWithConfig(opts Options, cfg Config) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts, cfg: cfg}
}

// ID returns the provider identifier ("anthropic" by default, overridable
// via Config.ProviderID for derived providers like Vertex Anthropic).
func (p *Provider) ID() string {
	return p.cfg.providerID()
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return &AnthropicModel{
		id:       modelID,
		provider: p,
	}
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns nil as Anthropic doesn't support image generation.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return nil
}

// EmbeddingModel returns nil as Anthropic doesn't support embeddings.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return nil
}

// SpeechModel returns nil as Anthropic doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns nil as Anthropic doesn't support transcription.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return nil
}

// RerankingModel returns nil as Anthropic doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

// Models returns available model IDs.
// Mirrors ai-sdk's AnthropicMessagesModelId union in
// packages/anthropic/src/anthropic-messages-options.ts.
func (p *Provider) Models() []string {
	return []string{
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-opus-4-5-20251101",
		"claude-opus-4-1",
		"claude-opus-4-1-20250805",
		"claude-opus-4-0",
		"claude-opus-4-20250514",
		"claude-sonnet-4-6",
		"claude-sonnet-4-5",
		"claude-sonnet-4-5-20250929",
		"claude-sonnet-4-0",
		"claude-sonnet-4-20250514",
		"claude-haiku-4-5",
		"claude-haiku-4-5-20251001",
		"claude-3-7-sonnet-20250219",
		"claude-3-5-sonnet-20241022",
		"claude-3-5-haiku-20241022",
		"claude-3-haiku-20240307",
	}
}
