// Package groq provides a Groq provider implementation.
// Groq uses an OpenAI-compatible API.
package groq

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

// groqCallWarner emits Groq chat unsupported-option warnings.
// Mirrors ai-sdk packages/groq/src/groq-chat-language-model.ts.
func groqCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	return warnings
}

const (
	defaultBaseURL = "https://api.groq.com/openai/v1"
)

// Options contains configuration for the Groq provider.
type Options struct {
	// APIKey is the Groq API key.
	APIKey string

	// BaseURL overrides the default API endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Groq provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new Groq provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:                "groq",
			BaseURL:                   baseURL,
			APIKey:                    opts.APIKey,
			Headers:                   opts.Headers,
			RequestModifier:           groqRequestModifier,
			CallWarner:                groqCallWarner,
			SupportsStructuredOutputs: true,
		}),
	}
}

// groqRequestModifier applies Groq-specific options to the request.
func groqRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	extra := make(map[string]any)

	if opts.ReasoningFormat != "" {
		extra["reasoning_format"] = opts.ReasoningFormat
	}
	if opts.ReasoningEffort != "" {
		extra["reasoning_effort"] = opts.ReasoningEffort
	}
	if opts.ServiceTier != "" {
		extra["service_tier"] = opts.ServiceTier
	}
	if opts.User != "" {
		extra["user"] = opts.User
	}
	if opts.ParallelToolCalls != nil {
		extra["parallel_tool_calls"] = *opts.ParallelToolCalls
	}

	return extra, nil, nil
}

// ID returns "groq".
func (p *Provider) ID() string {
	return "groq"
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

// LanguageModel returns a language model instance.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.Model(modelID)
}

// ImageModel returns nil as Groq doesn't support image generation.
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return nil
}

// EmbeddingModel returns nil as Groq doesn't support embeddings.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return nil
}

// SpeechModel returns nil as Groq doesn't support speech generation.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return nil
}

// TranscriptionModel returns a transcription model instance.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &GroqTranscriptionModel{
		id:       modelID,
		provider: p,
	}
}

// RerankingModel returns nil as Groq doesn't support reranking.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return nil
}

// Models returns available model IDs. Mirrors ai-sdk's GroqChatModelId
// union in packages/groq/src/groq-chat-options.ts.
func (p *Provider) Models() []string {
	return []string{
		// production models
		"gemma2-9b-it",
		"llama-3.1-8b-instant",
		"llama-3.3-70b-versatile",
		"meta-llama/llama-guard-4-12b",
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b",
		// preview models
		"deepseek-r1-distill-llama-70b",
		"meta-llama/llama-4-maverick-17b-128e-instruct",
		"meta-llama/llama-4-scout-17b-16e-instruct",
		"meta-llama/llama-prompt-guard-2-22m",
		"meta-llama/llama-prompt-guard-2-86m",
		"moonshotai/kimi-k2-instruct-0905",
		"qwen/qwen3-32b",
		"llama-guard-3-8b",
		"llama3-70b-8192",
		"llama3-8b-8192",
		"mixtral-8x7b-32768",
		"qwen-qwq-32b",
		"qwen-2.5-32b",
		"deepseek-r1-distill-qwen-32b",
	}
}

// TranscriptionModels returns available transcription model IDs.
func (p *Provider) TranscriptionModels() []string {
	return []string{
		"whisper-large-v3",
		"whisper-large-v3-turbo",
		"distil-whisper-large-v3-en",
	}
}

// Ensure Provider implements the provider interface
var _ provider.Provider = (*Provider)(nil)
