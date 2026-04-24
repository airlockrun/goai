package xai

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

func hasWarningFeature(warnings []stream.Warning, feature string) bool {
	for _, w := range warnings {
		if w.Feature == feature {
			return true
		}
	}
	return false
}

func TestXai_ChatCallWarner(t *testing.T) {
	tests := []struct {
		name    string
		opts    *stream.CallOptions
		feature string
	}{
		{
			name:    "topK",
			opts:    &stream.CallOptions{Messages: basicMsgs(), TopK: intPtr(10)},
			feature: "topK",
		},
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
			name:    "stopSequences",
			opts:    &stream.CallOptions{Messages: basicMsgs(), StopSequences: []string{"STOP"}},
			feature: "stopSequences",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warnings := xaiChatCallWarner(tc.opts)
			if !hasWarningFeature(warnings, tc.feature) {
				t.Errorf("expected %s warning, got %+v", tc.feature, warnings)
			}
		})
	}
}

func TestXaiResponses_BuildRequest_Warnings(t *testing.T) {
	p := New(Options{APIKey: "k"})
	m := p.Responses("grok-4").(*XaiResponsesModel)

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages:         basicMsgs(),
		StopSequences:    []string{"STOP"},
		FrequencyPenalty: floatPtr(0.5),
		PresencePenalty:  floatPtr(0.5),
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, feature := range []string{"stopSequences", "frequencyPenalty", "presencePenalty"} {
		if !hasWarningFeature(warnings, feature) {
			t.Errorf("expected %s warning, got %+v", feature, warnings)
		}
	}
}

func TestXaiResponses_UnknownProviderTool(t *testing.T) {
	p := New(Options{APIKey: "k"})
	m := p.Responses("grok-4").(*XaiResponsesModel)

	unknownTool := tool.Tool{
		Type:       "provider",
		ProviderID: "some.unknown.tool",
	}
	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: basicMsgs(),
		Tools:    []tool.Tool{unknownTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !hasWarningFeature(warnings, "tool") {
		t.Errorf("expected unknown-tool warning, got %+v", warnings)
	}
}

func TestXaiResponses_ForcedHostedToolChoice(t *testing.T) {
	p := New(Options{APIKey: "k"})
	m := p.Responses("grok-4").(*XaiResponsesModel)

	hosted := WebSearch()
	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages:   basicMsgs(),
		Tools:      []tool.Tool{hosted},
		ToolChoice: map[string]any{"type": "tool", "toolName": hosted.ProviderID},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !hasWarningFeature(warnings, "toolChoice") {
		t.Errorf("expected toolChoice warning, got %+v", warnings)
	}
}
