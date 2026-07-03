package bedrock

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }

func basicMsgs() []message.Message {
	return []message.Message{message.NewUserMessage("hi")}
}

func hasBedrockWarningFeature(warnings []stream.Warning, feature string) bool {
	for _, w := range warnings {
		if w.Feature == feature {
			return true
		}
	}
	return false
}

func TestBedrockAnthropic_BuildRequest_Warnings(t *testing.T) {
	p := New(Options{
		AccessKeyID:     "k",
		SecretAccessKey: "s",
		Region:          "us-east-1",
	})
	m := p.Model("anthropic.claude-3-sonnet-20240229-v1:0").(*BedrockLanguageModel)

	tests := []struct {
		name    string
		opts    *stream.CallOptions
		feature string
	}{
		{
			name:    "frequencyPenalty",
			opts:    &stream.CallOptions{Messages: basicMsgs(), FrequencyPenalty: floatPtr(0.5)},
			feature: "frequencyPenalty",
		},
		{
			name:    "presencePenalty",
			opts:    &stream.CallOptions{Messages: basicMsgs(), PresencePenalty: floatPtr(0.5)},
			feature: "presencePenalty",
		},
		{
			name:    "seed",
			opts:    &stream.CallOptions{Messages: basicMsgs(), Seed: intPtr(42)},
			feature: "seed",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, warnings, err := m.buildAnthropicRequest(tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if !hasBedrockWarningFeature(warnings, tc.feature) {
				t.Errorf("expected %s warning, got %+v", tc.feature, warnings)
			}
		})
	}
}

func TestBedrockAnthropic_UnknownProviderTool(t *testing.T) {
	p := New(Options{AccessKeyID: "k", SecretAccessKey: "s", Region: "us-east-1"})
	m := p.Model("anthropic.claude-3-sonnet-20240229-v1:0").(*BedrockLanguageModel)

	unknownTool := tool.Tool{
		Type:       "provider",
		ProviderID: "some.unknown.tool",
	}
	_, warnings, err := m.buildAnthropicRequest(&stream.CallOptions{
		Messages: basicMsgs(),
		Tools:    []tool.Tool{unknownTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !hasBedrockWarningFeature(warnings, "tool") {
		t.Errorf("expected unknown-tool warning, got %+v", warnings)
	}
}
