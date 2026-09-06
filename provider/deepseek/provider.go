// Package deepseek provides a DeepSeek provider implementation.
// DeepSeek uses an OpenAI-compatible API.
package deepseek

import (
	"fmt"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.deepseek.com/v1"
)

// Options contains configuration for the DeepSeek provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the DeepSeek provider.
type Provider struct {
	compat *openaicompat.Provider
}

// New creates a new DeepSeek provider.
func New(opts Options) *Provider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Provider{
		compat: openaicompat.New(openaicompat.Options{
			ProviderID:      "deepseek",
			BaseURL:         baseURL,
			APIKey:          opts.APIKey,
			Headers:         opts.Headers,
			RequestModifier: deepseekRequestModifier,
			CallWarner:      deepseekCallWarner,
			ModelCallWarner: func(id string, options *stream.CallOptions) []stream.Warning {
				opts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
				if err != nil {
					return nil
				}
				thinking := (id == "deepseek-reasoner" || strings.Contains(id, "deepseek-v4") || opts.Thinking != nil || options.Reasoning != "" && options.Reasoning != "provider-default") && options.Reasoning != "none"
				if opts.Thinking != nil {
					thinking = opts.Thinking.Type != "disabled"
				}
				var warnings []stream.Warning
				if thinking && options.Temperature != nil {
					warnings = append(warnings, stream.UnsupportedWarning("temperature", "thinking is enabled"))
				}
				if thinking && options.TopP != nil {
					warnings = append(warnings, stream.UnsupportedWarning("topP", "thinking is enabled"))
				}
				return warnings
			},
			MessageConverter: convertMessages,
			TransformRequest: func(id string, body map[string]any) error {
				thinking, _ := body["thinking"].(map[string]any)
				if effort, ok := body["reasoning_effort"].(string); ok {
					if thinking == nil {
						typeName := "enabled"
						if effort == "none" {
							typeName = "disabled"
						}
						thinking = map[string]any{"type": typeName}
						body["thinking"] = thinking
					}
					switch effort {
					case "minimal":
						body["reasoning_effort"] = "low"
					case "medium":
						body["reasoning_effort"] = "high"
					case "xhigh":
						body["reasoning_effort"] = "max"
					}
					if thinking["type"] == "disabled" || effort == "none" {
						delete(body, "reasoning_effort")
					}
				}
				if thinking["type"] != "disabled" && (thinking != nil || id == "deepseek-reasoner" || strings.Contains(id, "deepseek-v4")) {
					delete(body, "temperature")
					delete(body, "top_p")
				}
				return nil
			},
		}),
	}
}

// deepseekRequestModifier applies DeepSeek-specific options to the request.
func deepseekRequestModifier(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](providerOptions)
	if err != nil {
		return nil, nil, err
	}

	extra := make(map[string]any)
	var warnings []stream.Warning

	// DeepSeek's current API accepts the same typed thinking object exposed by
	// ai-sdk. Leaving it unset preserves the provider default.
	if opts.Thinking != nil {
		switch opts.Thinking.Type {
		case "adaptive":
			extra["thinking"] = map[string]any{"type": "enabled"}
			warnings = append(warnings, stream.UnsupportedWarning("thinking.type", "adaptive is mapped to enabled"))
		case "enabled", "disabled":
			extra["thinking"] = map[string]any{"type": opts.Thinking.Type}
		default:
			return nil, nil, fmt.Errorf("unsupported DeepSeek thinking type %q", opts.Thinking.Type)
		}
	}

	// reasoning_effort tunes thinking strength on V4 reasoning models. It is
	// suppressed when thinking is disabled, matching ai-sdk #15235.
	if opts.ReasoningEffort != "" && !(opts.Thinking != nil && opts.Thinking.Type == "disabled") {
		effort := opts.ReasoningEffort
		switch effort {
		case "medium":
			effort = "high"
		case "xhigh":
			effort = "max"
		case "low", "high", "max":
		default:
			return nil, nil, fmt.Errorf("unsupported DeepSeek reasoning effort %q", effort)
		}
		if effort != opts.ReasoningEffort {
			warnings = append(warnings, stream.UnsupportedWarning("reasoningEffort", "mapped to "+effort))
		}
		extra["reasoning_effort"] = effort
	}

	return extra, warnings, nil
}

// deepseekCallWarner emits DeepSeek chat unsupported-option warnings.
func deepseekCallWarner(options *stream.CallOptions) []stream.Warning {
	var warnings []stream.Warning
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}
	if options.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", "not supported by DeepSeek"))
	}
	if options.PresencePenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", "not supported by DeepSeek"))
	}
	return warnings
}

func (p *Provider) ID() string                                                 { return "deepseek" }
func (p *Provider) Model(modelID string) stream.Model                          { return p.compat.Model(modelID) }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel           { return p.Model(modelID) }
func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)
