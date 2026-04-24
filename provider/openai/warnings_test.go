package openai

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func intPtr(v int) *int { return &v }

func basicOpenAIMsgs() []message.Message {
	return []message.Message{message.NewUserMessage("hi")}
}

func hasOpenAIWarning(warnings []stream.Warning, feature string, wantType stream.WarningType) bool {
	for _, w := range warnings {
		if w.Type == wantType && w.Feature == feature {
			return true
		}
	}
	return false
}

func TestOpenAIChat_BuildRequest_TopKWarning(t *testing.T) {
	p := New(provider.Options{APIKey: "k"}).Chat("gpt-4o")
	m := p.(*ChatModel)
	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: basicOpenAIMsgs(),
		TopK:     intPtr(10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !hasOpenAIWarning(warnings, "topK", stream.WarningUnsupported) {
		t.Errorf("expected topK warning, got %+v", warnings)
	}
}

func TestOpenAIResponses_BuildRequest_Warnings(t *testing.T) {
	p := New(provider.Options{APIKey: "k"}).Responses("gpt-4o")
	m := p.(*ResponsesModel)

	topK := 5
	seed := 42
	pen := 0.5
	freq := 0.5

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages:         basicOpenAIMsgs(),
		TopK:             &topK,
		Seed:             &seed,
		PresencePenalty:  &pen,
		FrequencyPenalty: &freq,
		StopSequences:    []string{"STOP"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, feature := range []string{"topK", "seed", "presencePenalty", "frequencyPenalty", "stopSequences"} {
		if !hasOpenAIWarning(warnings, feature, stream.WarningUnsupported) {
			t.Errorf("expected %s warning, got %+v", feature, warnings)
		}
	}
}

func TestOpenAIResponses_BuildRequest_UnknownProviderTool(t *testing.T) {
	p := New(provider.Options{APIKey: "k"}).Responses("gpt-4o")
	m := p.(*ResponsesModel)

	unknownTool := tool.Tool{
		Type:       "provider",
		ProviderID: "some.unknown.tool",
	}
	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: basicOpenAIMsgs(),
		Tools:    []tool.Tool{unknownTool},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !hasOpenAIWarning(warnings, "tool", stream.WarningUnsupported) {
		t.Errorf("expected unknown-tool warning, got %+v", warnings)
	}
}
