package mistral

import (
	"testing"

	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int           { return &v }
func floatPtr(v float64) *float64 { return &v }

func hasWarningFeature(warnings []stream.Warning, feature string) bool {
	for _, w := range warnings {
		if w.Feature == feature {
			return true
		}
	}
	return false
}

func TestMistralCallWarner(t *testing.T) {
	tests := []struct {
		name    string
		opts    *stream.CallOptions
		feature string
	}{
		{
			name:    "topK",
			opts:    &stream.CallOptions{TopK: intPtr(5)},
			feature: "topK",
		},
		{
			name:    "frequencyPenalty",
			opts:    &stream.CallOptions{FrequencyPenalty: floatPtr(0.5)},
			feature: "frequencyPenalty",
		},
		{
			name:    "presencePenalty",
			opts:    &stream.CallOptions{PresencePenalty: floatPtr(0.5)},
			feature: "presencePenalty",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warnings := mistralCallWarner(tc.opts)
			if !hasWarningFeature(warnings, tc.feature) {
				t.Errorf("expected %s warning, got %+v", tc.feature, warnings)
			}
		})
	}
}
