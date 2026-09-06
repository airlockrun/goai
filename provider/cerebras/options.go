package cerebras

import (
	"errors"
	"fmt"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

type ChatOptions struct {
	User              string             `json:"user,omitempty"`
	StrictJSONSchema  *bool              `json:"strictJsonSchema,omitempty"`
	ParallelToolCalls *bool              `json:"parallelToolCalls,omitempty"`
	Logprobs          *bool              `json:"logprobs,omitempty"`
	TopLogprobs       *int               `json:"topLogprobs,omitempty"`
	LogitBias         map[string]float64 `json:"logitBias,omitempty"`
	ServiceTier       string             `json:"serviceTier,omitempty"`
	ReasoningEffort   string             `json:"reasoningEffort,omitempty"`
	ReasoningFormat   string             `json:"reasoningFormat,omitempty"`
	Prediction        *Prediction        `json:"prediction,omitempty"`
	PromptCacheKey    string             `json:"promptCacheKey,omitempty"`
}

type Prediction struct {
	Type    string `json:"type"`
	Content any    `json:"content"`
}

func cerebrasRequestModifier(values map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](values)
	if err != nil {
		return nil, nil, err
	}
	if opts.TopLogprobs != nil && (*opts.TopLogprobs < 0 || *opts.TopLogprobs > 20) {
		return nil, nil, errors.New("topLogprobs must be between 0 and 20")
	}
	for _, value := range opts.LogitBias {
		if value < -100 || value > 100 {
			return nil, nil, errors.New("logitBias must be between -100 and 100")
		}
	}
	if len(opts.PromptCacheKey) > 1024 {
		return nil, nil, errors.New("promptCacheKey exceeds 1024 bytes")
	}
	for name, value := range map[string]string{"serviceTier": opts.ServiceTier, "reasoningEffort": opts.ReasoningEffort, "reasoningFormat": opts.ReasoningFormat} {
		allowed := map[string][]string{"serviceTier": {"", "auto", "default", "flex", "priority"}, "reasoningEffort": {"", "none", "low", "medium", "high"}, "reasoningFormat": {"", "none", "parsed", "text_parsed", "raw", "hidden"}}
		valid := false
		for _, candidate := range allowed[name] {
			if candidate == value {
				valid = true
			}
		}
		if !valid {
			return nil, nil, fmt.Errorf("invalid %s %q", name, value)
		}
	}
	extra := map[string]any{}
	for key, value := range map[string]string{"user": opts.User, "service_tier": opts.ServiceTier, "reasoning_effort": opts.ReasoningEffort, "reasoning_format": opts.ReasoningFormat, "prompt_cache_key": opts.PromptCacheKey} {
		if value != "" {
			extra[key] = value
		}
	}
	if opts.ParallelToolCalls != nil {
		extra["parallel_tool_calls"] = *opts.ParallelToolCalls
	}
	if opts.Logprobs != nil {
		extra["logprobs"] = *opts.Logprobs
	}
	if opts.TopLogprobs != nil {
		extra["top_logprobs"] = *opts.TopLogprobs
	}
	if opts.LogitBias != nil {
		extra["logit_bias"] = opts.LogitBias
	}
	if opts.Prediction != nil {
		extra["prediction"] = opts.Prediction
	}
	return extra, nil, nil
}
