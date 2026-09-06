// Package openai provides an OpenAI provider implementation.
//
// This package follows the same pattern as @ai-sdk/openai, providing both:
//   - Chat Completions API via Chat() method
//   - Responses API via Responses() method
//
// The default Model() method returns a Responses API model, matching @ai-sdk behavior.
package openai

import (
	"net/http"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
)

// Provider implements the OpenAI provider.
type Provider struct {
	opts provider.Options
}

// New creates a new OpenAI provider.
func New(opts provider.Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

// ID returns "openai".
func (p *Provider) ID() string {
	return "openai"
}

// Model returns a model instance using the Responses API (default).
// This matches @ai-sdk/openai behavior where the default is the Responses API.
func (p *Provider) Model(modelID string) stream.Model {
	return p.Responses(modelID)
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Responses(modelID)
}

// Chat returns a model instance using the Chat Completions API (/chat/completions).
func (p *Provider) Chat(modelID string) stream.Model {
	return &ChatModel{
		id:       modelID,
		provider: p,
	}
}

// Responses returns a model instance using the Responses API (/responses).
func (p *Provider) Responses(modelID string) stream.Model {
	headers := map[string]string{"Authorization": "Bearer " + p.opts.APIKey}
	if p.opts.Organization != "" {
		headers["OpenAI-Organization"] = p.opts.Organization
	}
	if p.opts.Project != "" {
		headers["OpenAI-Project"] = p.opts.Project
	}
	for k, v := range p.opts.Headers {
		headers[http.CanonicalHeaderKey(k)] = v
	}
	return NewResponsesModel(modelID, ResponsesConfig{
		Provider: "openai.responses", URL: strings.TrimRight(p.opts.BaseURL, "/") + "/responses", Headers: headers,
		ConfigureRequest: func(*http.Request) error { return nil },
	})
}

// ImageModel returns an image generation model instance.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &OpenAIImageModel{
		id:       modelID,
		provider: p,
	}
}

// EmbeddingModel returns an embedding model instance.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &OpenAIEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

// SpeechModel returns a text-to-speech model using the dedicated /audio/speech
// endpoint. Multimodal chat-audio models (gpt-audio) have no such endpoint —
// callers route those to ChatSpeechModel instead. That decision needs model
// modality data, which lives in the catalog (sol), not in this provider, so
// SpeechModel does NOT auto-detect; it always returns the dedicated model.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &OpenAISpeechModel{
		id:       modelID,
		provider: p,
	}
}

// ChatSpeechModel returns a speech model backed by /chat/completions (modalities
// + audio), for multimodal chat-audio models that produce speech through chat
// rather than a dedicated /audio/speech endpoint.
func (p *Provider) ChatSpeechModel(modelID string) model.SpeechModel {
	return &chatSpeechModel{id: modelID, provider: p}
}

// TranscriptionModel returns a speech-to-text model using the dedicated
// /audio/transcriptions endpoint. Chat-audio models use ChatTranscriptionModel,
// selected by the caller from modality data (see SpeechModel).
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &OpenAITranscriptionModel{
		id:       modelID,
		provider: p,
	}
}

// ChatTranscriptionModel returns a transcription model backed by
// /chat/completions, for chat-audio models with no /audio/transcriptions
// endpoint. These models are conversational, so any instruction to make them
// transcribe rather than reply is application policy and must be supplied by the
// caller via TranscribeCallOptions.Prompt — goai bakes in none.
func (p *Provider) ChatTranscriptionModel(modelID string) model.TranscriptionModel {
	return &chatTranscriptionModel{id: modelID, provider: p}
}

// RerankingModel returns nil as OpenAI doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}
