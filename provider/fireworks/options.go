package fireworks

import (
	"errors"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

type ChatOptions struct {
	PromptCacheKey   string           `json:"promptCacheKey,omitempty"`
	ServiceTier      string           `json:"serviceTier,omitempty"`
	Thinking         *ThinkingOptions `json:"thinking,omitempty"`
	ReasoningHistory string           `json:"reasoningHistory,omitempty"`
	ReasoningEffort  string           `json:"reasoningEffort,omitempty"`
}
type ThinkingOptions struct {
	Type         string `json:"type,omitempty"`
	BudgetTokens *int   `json:"budgetTokens,omitempty"`
}

func fireworksRequestModifier(values map[string]any) (map[string]any, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](values)
	if err != nil {
		return nil, nil, err
	}
	if opts.ServiceTier != "" && opts.ServiceTier != "priority" {
		return nil, nil, errors.New("invalid Fireworks service tier")
	}
	switch opts.ReasoningHistory {
	case "", "disabled", "interleaved", "preserved":
	default:
		return nil, nil, errors.New("invalid Fireworks reasoning history")
	}
	body := map[string]any{}
	for key, value := range map[string]string{"prompt_cache_key": opts.PromptCacheKey, "service_tier": opts.ServiceTier, "reasoning_history": opts.ReasoningHistory} {
		if value != "" {
			body[key] = value
		}
	}
	if opts.Thinking != nil {
		thinking := map[string]any{}
		switch opts.Thinking.Type {
		case "":
		case "enabled", "disabled":
			thinking["type"] = opts.Thinking.Type
		default:
			return nil, nil, errors.New("invalid Fireworks thinking type")
		}
		if opts.Thinking.BudgetTokens != nil {
			if *opts.Thinking.BudgetTokens < 1024 {
				return nil, nil, errors.New("Fireworks thinking budget must be at least 1024")
			}
			thinking["budget_tokens"] = *opts.Thinking.BudgetTokens
		}
		body["thinking"] = thinking
	}
	if opts.ReasoningEffort != "" {
		effort := opts.ReasoningEffort
		if effort == "minimal" {
			effort = "low"
		}
		if effort == "xhigh" {
			effort = "high"
		}
		body["reasoning_effort"] = effort
	}
	return body, nil, nil
}
