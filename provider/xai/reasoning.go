package xai

import (
	"regexp"

	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

var modelsWithoutReasoningEffort = regexp.MustCompile(`^grok-4\.20(-\d{4})?-(non-)?reasoning$`)

func reasoningEffort(id, reasoning, explicit string) (string, []stream.Warning) {
	if explicit != "" {
		return explicit, nil
	}
	values := map[string]string{"none": "none", "minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}
	if id == "grok-4.6" {
		values["xhigh"] = "xhigh"
	}
	if modelsWithoutReasoningEffort.MatchString(id) {
		values = nil
	}
	return provider.MapReasoning(reasoning, values)
}
