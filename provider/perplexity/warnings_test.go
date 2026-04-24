package perplexity

import (
	"testing"

	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int { return &v }

func hasWarningFeature(warnings []stream.Warning, feature string) bool {
	for _, w := range warnings {
		if w.Feature == feature {
			return true
		}
	}
	return false
}

func TestPerplexityCallWarner(t *testing.T) {
	tests := []struct {
		name    string
		opts    *stream.CallOptions
		feature string
	}{
		{
			name:    "topK",
			opts:    &stream.CallOptions{TopK: intPtr(10)},
			feature: "topK",
		},
		{
			name:    "stopSequences",
			opts:    &stream.CallOptions{StopSequences: []string{"STOP"}},
			feature: "stopSequences",
		},
		{
			name:    "seed",
			opts:    &stream.CallOptions{Seed: intPtr(42)},
			feature: "seed",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warnings := perplexityCallWarner(tc.opts)
			if !hasWarningFeature(warnings, tc.feature) {
				t.Errorf("expected %s warning, got %+v", tc.feature, warnings)
			}
		})
	}
}
