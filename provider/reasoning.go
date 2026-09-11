package provider

import (
	"fmt"
	"math"

	"github.com/airlockrun/goai/stream"
)

// MapReasoning maps shared effort to supported wire values and reports lossy mappings.
// Empty and provider-default leave the provider's defaults untouched.
func MapReasoning(effort string, values map[string]string) (string, []stream.Warning) {
	if effort == "" || effort == "provider-default" {
		return "", nil
	}
	mapped := values[effort]
	if mapped == "" {
		return "", []stream.Warning{stream.UnsupportedWarning("reasoning", fmt.Sprintf("reasoning %q is not supported by this model", effort))}
	}
	if mapped != effort {
		return mapped, []stream.Warning{stream.CompatibilityWarning("reasoning", fmt.Sprintf("reasoning %q is mapped to %q", effort, mapped))}
	}
	return mapped, nil
}

// ReasoningBudget derives a token budget from the model output limit.
func ReasoningBudget(effort string, limit int) (int, []stream.Warning) {
	ratio, ok := map[string]float64{"minimal": .02, "low": .1, "medium": .3, "high": .6, "xhigh": .9}[effort]
	if !ok {
		_, warnings := MapReasoning(effort, nil)
		return 0, warnings
	}
	return min(limit, max(1024, int(math.Round(float64(limit)*ratio)))), nil
}

// OpenAIReasoning resolves explicit effort before the shared setting.
func OpenAIReasoning(effort, explicit string) (string, []stream.Warning) {
	if explicit != "" {
		return explicit, nil
	}
	return MapReasoning(effort, map[string]string{"none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"})
}
