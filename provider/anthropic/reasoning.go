package anthropic

import (
	"strings"

	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// ReasoningConfiguration derives Claude thinking and effort settings. Provider
// adapters merge explicit options over these values before building requests.
func ReasoningConfiguration(modelID, reasoning string) (*ThinkingConfig, string, []stream.Warning) {
	if reasoning == "" || reasoning == "provider-default" {
		return nil, "", nil
	}
	if reasoning == "none" {
		return &ThinkingConfig{Type: "disabled"}, "", nil
	}
	limit, _, _ := modelSupport(modelID)
	if limit == 128000 {
		values := map[string]string{"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
		if strings.Contains(modelID, "claude-sonnet-4-6") || strings.Contains(modelID, "claude-opus-4-6") {
			values["xhigh"] = "max"
		}
		effort, warnings := provider.MapReasoning(reasoning, values)
		if effort == "" {
			return nil, "", warnings
		}
		return &ThinkingConfig{Type: "adaptive", Display: "summarized"}, effort, warnings
	}
	budget, warnings := provider.ReasoningBudget(reasoning, limit)
	if budget == 0 {
		return nil, "", warnings
	}
	return &ThinkingConfig{Type: "enabled", BudgetTokens: budget}, "", warnings
}
