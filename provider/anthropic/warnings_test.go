package anthropic

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }

func basicAnthropicMsgs() []message.Message {
	return []message.Message{message.NewUserMessage("hi")}
}

func newAnthropicTestModel() *AnthropicModel {
	p := New(Options{APIKey: "k"})
	return p.LanguageModel("claude-sonnet-4-5-20250929").(*AnthropicModel)
}

func hasWarning(warnings []stream.Warning, want stream.Warning) bool {
	for _, w := range warnings {
		if w.Type == want.Type && w.Feature == want.Feature && w.Message == want.Message {
			return true
		}
	}
	return false
}

func TestAnthropic_BuildRequest_Warnings(t *testing.T) {
	m := newAnthropicTestModel()

	tests := []struct {
		name    string
		opts    *stream.CallOptions
		want    stream.Warning
		feature string
	}{
		{
			name: "frequencyPenalty",
			opts: &stream.CallOptions{
				Messages:         basicAnthropicMsgs(),
				FrequencyPenalty: floatPtr(0.5),
			},
			want: stream.UnsupportedWarning("frequencyPenalty", ""),
		},
		{
			name: "presencePenalty",
			opts: &stream.CallOptions{
				Messages:        basicAnthropicMsgs(),
				PresencePenalty: floatPtr(0.5),
			},
			want: stream.UnsupportedWarning("presencePenalty", ""),
		},
		{
			name: "seed",
			opts: &stream.CallOptions{
				Messages: basicAnthropicMsgs(),
				Seed:     intPtr(42),
			},
			want: stream.UnsupportedWarning("seed", ""),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, warnings, err := m.buildRequest(tc.opts)
			if err != nil {
				t.Fatalf("buildRequest: %v", err)
			}
			if !hasWarning(warnings, tc.want) {
				t.Errorf("missing warning %+v in %+v", tc.want, warnings)
			}
		})
	}
}

func TestAnthropic_BuildRequest_TemperatureClamping(t *testing.T) {
	m := newAnthropicTestModel()

	t.Run("above maximum clamped with warning", func(t *testing.T) {
		_, warnings, err := m.buildRequest(&stream.CallOptions{
			Messages:    basicAnthropicMsgs(),
			Temperature: floatPtr(1.5),
		})
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, w := range warnings {
			if w.Type == stream.WarningUnsupported && w.Feature == "temperature" {
				found = true
			}
		}
		if !found {
			t.Errorf("expected temperature clamp warning, got %+v", warnings)
		}
	})

	t.Run("below minimum clamped with warning", func(t *testing.T) {
		_, warnings, err := m.buildRequest(&stream.CallOptions{
			Messages:    basicAnthropicMsgs(),
			Temperature: floatPtr(-0.5),
		})
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, w := range warnings {
			if w.Type == stream.WarningUnsupported && w.Feature == "temperature" {
				found = true
			}
		}
		if !found {
			t.Errorf("expected temperature clamp warning, got %+v", warnings)
		}
	})
}

func TestAnthropic_BuildRequest_ResponseFormatJSONNoSchema(t *testing.T) {
	m := newAnthropicTestModel()

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages:       basicAnthropicMsgs(),
		ResponseFormat: &stream.ResponseFormat{Type: "json"},
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range warnings {
		if w.Type == stream.WarningUnsupported && w.Feature == "responseFormat" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected responseFormat warning, got %+v", warnings)
	}
}

func TestAnthropic_BuildRequest_UnknownProviderTool(t *testing.T) {
	m := newAnthropicTestModel()

	unknownTool := tool.Tool{
		Type:       "provider",
		ProviderID: "some.unknown.tool",
	}
	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: basicAnthropicMsgs(),
		Tools:    []tool.Tool{unknownTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range warnings {
		if w.Type == stream.WarningUnsupported && w.Feature == "tool" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected unknown-tool warning, got %+v", warnings)
	}
}
