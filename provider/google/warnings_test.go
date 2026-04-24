package google

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func floatPtr(v float64) *float64 { return &v }

func newGoogleTestModel() *GoogleModel {
	p := New(Options{APIKey: "k"})
	return p.Model("gemini-2.5-pro").(*GoogleModel)
}

func hasGoogleWarning(warnings []stream.Warning, feature string, wantType stream.WarningType) bool {
	for _, w := range warnings {
		if w.Type == wantType && w.Feature == feature {
			return true
		}
	}
	return false
}

func TestGoogle_BuildRequest_Warnings(t *testing.T) {
	m := newGoogleTestModel()

	tests := []struct {
		name    string
		opts    *stream.CallOptions
		feature string
	}{
		{
			name: "frequencyPenalty",
			opts: &stream.CallOptions{
				Messages:         []message.Message{message.NewUserMessage("hi")},
				FrequencyPenalty: floatPtr(0.5),
			},
			feature: "frequencyPenalty",
		},
		{
			name: "presencePenalty",
			opts: &stream.CallOptions{
				Messages:        []message.Message{message.NewUserMessage("hi")},
				PresencePenalty: floatPtr(0.5),
			},
			feature: "presencePenalty",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, warnings, err := m.buildRequest(tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if !hasGoogleWarning(warnings, tc.feature, stream.WarningUnsupported) {
				t.Errorf("expected %s warning, got %+v", tc.feature, warnings)
			}
		})
	}
}
