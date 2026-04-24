// Package togetherai provides a Together AI provider implementation.
// Together AI uses an OpenAI-compatible API.
package togetherai

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.together.xyz/v1"
)

// Options contains configuration for the Together AI provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Together AI provider.
type Provider struct {
	compat *openaicompat.Provider
	opts   Options
}

// New creates a new Together AI provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		opts: opts,
		compat: openaicompat.New(openaicompat.Options{
			ProviderID: "togetherai",
			BaseURL:    baseURL,
			APIKey:     opts.APIKey,
			Headers:    opts.Headers,
		}),
	}
}

func (p *Provider) ID() string                                                 { return "togetherai" }
func (p *Provider) Model(modelID string) stream.Model                          { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel           { return p.Model(modelID) }
func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

// Models returns the Together AI chat-model catalog. Mirrors ai-sdk's
// TogetherAIChatModelId union in packages/togetherai/src/togetherai-chat-options.ts.
func (p *Provider) Models() []string {
	return []string{
		// Meta Llama Turbo / Lite / chat variants
		"meta-llama/Llama-3.3-70B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
		"meta-llama/Llama-3.2-3B-Instruct-Turbo",
		"meta-llama/Meta-Llama-3-8B-Instruct-Lite",
		"meta-llama/Meta-Llama-3-70B-Instruct-Lite",
		"meta-llama/Llama-3-8b-chat-hf",
		"meta-llama/Llama-3-70b-chat-hf",
		"meta-llama/Llama-2-13b-chat-hf",
		// Nvidia / Qwen / Microsoft
		"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
		"Qwen/Qwen2.5-Coder-32B-Instruct",
		"Qwen/QwQ-32B-Preview",
		"Qwen/Qwen2.5-7B-Instruct-Turbo",
		"Qwen/Qwen2.5-72B-Instruct-Turbo",
		"Qwen/Qwen2-72B-Instruct",
		"microsoft/WizardLM-2-8x22B",
		// Google Gemma
		"google/gemma-2-27b-it",
		"google/gemma-2-9b-it",
		"google/gemma-2b-it",
		// Databricks / DeepSeek / Nous / Upstage / Gryphe
		"databricks/dbrx-instruct",
		"deepseek-ai/deepseek-llm-67b-chat",
		"deepseek-ai/DeepSeek-V3",
		"NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
		"upstage/SOLAR-10.7B-Instruct-v1.0",
		"Gryphe/MythoMax-L2-13b",
		// Mistral
		"mistralai/Mistral-7B-Instruct-v0.1",
		"mistralai/Mistral-7B-Instruct-v0.2",
		"mistralai/Mistral-7B-Instruct-v0.3",
		"mistralai/Mixtral-8x7B-Instruct-v0.1",
		"mistralai/Mixtral-8x22B-Instruct-v0.1",
	}
}

var _ provider.Provider = (*Provider)(nil)
