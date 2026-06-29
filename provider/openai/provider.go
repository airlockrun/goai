// Package openai provides an OpenAI provider implementation.
//
// This package follows the same pattern as @ai-sdk/openai, providing both:
//   - Chat Completions API via Chat() method
//   - Responses API via Responses() method
//
// The default Model() method returns a Responses API model, matching @ai-sdk behavior.
package openai

import (
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
	return &ResponsesModel{
		id:       modelID,
		provider: p,
	}
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

// Models returns available model IDs.
// Mirrors ai-sdk's OpenAIChatModelId union in
// packages/openai/src/chat/openai-chat-options.ts.
func (p *Provider) Models() []string {
	return []string{
		// GPT-5.5 family (latest)
		"gpt-5.5",
		"gpt-5.5-2026-04-23",
		// GPT-5.4
		"gpt-5.4",
		"gpt-5.4-2026-03-05",
		"gpt-5.4-mini",
		"gpt-5.4-mini-2026-03-17",
		"gpt-5.4-nano",
		"gpt-5.4-nano-2026-03-17",
		"gpt-5.4-pro",
		"gpt-5.4-pro-2026-03-05",
		// GPT-5.3
		"gpt-5.3-chat-latest",
		"gpt-5.3-codex",
		// GPT-5.2
		"gpt-5.2",
		"gpt-5.2-2025-12-11",
		"gpt-5.2-chat-latest",
		"gpt-5.2-pro",
		"gpt-5.2-pro-2025-12-11",
		// GPT-5.1
		"gpt-5.1",
		"gpt-5.1-2025-11-13",
		"gpt-5.1-chat-latest",
		// GPT-5
		"gpt-5",
		"gpt-5-2025-08-07",
		"gpt-5-mini",
		"gpt-5-mini-2025-08-07",
		"gpt-5-nano",
		"gpt-5-nano-2025-08-07",
		"gpt-5-chat-latest",
		// GPT-4.1
		"gpt-4.1",
		"gpt-4.1-2025-04-14",
		"gpt-4.1-mini",
		"gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano",
		"gpt-4.1-nano-2025-04-14",
		// GPT-4o
		"gpt-4o",
		"gpt-4o-2024-05-13",
		"gpt-4o-2024-08-06",
		"gpt-4o-2024-11-20",
		"gpt-4o-audio-preview",
		"gpt-4o-audio-preview-2024-12-17",
		"gpt-4o-audio-preview-2025-06-03",
		"gpt-4o-mini",
		"gpt-4o-mini-2024-07-18",
		"gpt-4o-mini-audio-preview",
		"gpt-4o-mini-audio-preview-2024-12-17",
		"gpt-4o-search-preview",
		"gpt-4o-search-preview-2025-03-11",
		"gpt-4o-mini-search-preview",
		"gpt-4o-mini-search-preview-2025-03-11",
		// GPT-4
		"gpt-4-turbo",
		"gpt-4",
		// GPT-3.5
		"gpt-3.5-turbo",
		"gpt-3.5-turbo-0125",
		"gpt-3.5-turbo-1106",
		"gpt-3.5-turbo-16k",
		// Reasoning models
		"o1",
		"o1-2024-12-17",
		"o3",
		"o3-2025-04-16",
		"o3-mini",
		"o3-mini-2025-01-31",
		"o4-mini",
		"o4-mini-2025-04-16",
	}
}

// EmbeddingModels returns available embedding model IDs.
func (p *Provider) EmbeddingModels() []string {
	return []string{
		"text-embedding-3-small",
		"text-embedding-3-large",
		"text-embedding-ada-002",
	}
}

// ImageModels returns available image model IDs.
func (p *Provider) ImageModels() []string {
	return []string{
		"dall-e-3",
		"dall-e-2",
		"gpt-image-1",
	}
}

// SpeechModels returns available speech model IDs.
func (p *Provider) SpeechModels() []string {
	return []string{
		"tts-1",
		"tts-1-hd",
		"gpt-4o-mini-tts",
	}
}

// TranscriptionModels returns available transcription model IDs.
func (p *Provider) TranscriptionModels() []string {
	return []string{
		"whisper-1",
		"gpt-4o-transcribe",
		"gpt-4o-transcribe-diarize",
		"gpt-4o-mini-transcribe",
	}
}
