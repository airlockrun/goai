package xai

import "regexp"

var modelsWithoutReasoningEffort = regexp.MustCompile(`^grok-4\.20(-\d{4})?-(non-)?reasoning$`)
