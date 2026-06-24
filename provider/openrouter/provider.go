// Package openrouter implements the OpenRouter modalities whose wire format
// differs from OpenAI's, so they can't ride the embedded OpenAI provider:
//
//   - image generation — POST {base}/images (OpenAI uses /images/generations)
//   - transcription   — POST {base}/audio/transcriptions with a JSON base64
//     body (OpenAI uses multipart form-data)
//   - speech (TTS)    — POST {base}/audio/speech, which IS OpenAI-compatible
//     and is reused verbatim here for a single place to construct OpenRouter
//     modality models.
//
// Chat and embeddings are handled elsewhere (sol routes chat through
// openaicompat); the language/embedding/reranking getters return nil so a
// stray call fails loud instead of mis-routing.
package openrouter

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// DefaultBaseURL is OpenRouter's OpenAI-compatible API root.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// Provider serves OpenRouter's image, speech, and transcription models.
type Provider struct {
	opts provider.Options
}

// New creates a new OpenRouter provider.
func New(opts provider.Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = DefaultBaseURL
	}
	return &Provider{opts: opts}
}

// ID returns "openrouter".
func (p *Provider) ID() string { return "openrouter" }

// Model / LanguageModel / EmbeddingModel / RerankingModel: not served here.
// OpenRouter chat + embeddings are wired separately; return nil.
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

// ImageModel returns an OpenRouter image-generation model (POST /images).
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &imageModel{id: modelID, provider: p}
}

// SpeechModel returns an OpenRouter text-to-speech model (POST /audio/speech).
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &speechModel{id: modelID, provider: p}
}

// TranscriptionModel returns an OpenRouter speech-to-text model
// (POST /audio/transcriptions, JSON base64 body).
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &transcriptionModel{id: modelID, provider: p}
}

// Models returns no static list — OpenRouter's catalog is fetched from its API.
func (p *Provider) Models() []string { return nil }

// setHeaders applies auth + provider/request headers, shared by all three models.
func (p *Provider) setHeaders(h interface{ Set(string, string) }, reqHeaders map[string]string) {
	h.Set("Content-Type", "application/json")
	h.Set("Authorization", "Bearer "+p.opts.APIKey)
	for k, v := range p.opts.Headers {
		h.Set(k, v)
	}
	for k, v := range reqHeaders {
		h.Set(k, v)
	}
}

var _ provider.Provider = (*Provider)(nil)
