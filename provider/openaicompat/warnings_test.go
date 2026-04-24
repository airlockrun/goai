package openaicompat

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int { return &v }

func TestCompatModel_CallWarner_Integration(t *testing.T) {
	// CallWarner warnings should flow through buildRequest.
	callWarner := func(options *stream.CallOptions) []stream.Warning {
		if options.TopK != nil {
			return []stream.Warning{stream.UnsupportedWarning("topK", "")}
		}
		return nil
	}

	p := New(Options{
		ProviderID: "test",
		BaseURL:    "https://example.com",
		APIKey:     "k",
		CallWarner: callWarner,
	})
	m := p.Model("test-model").(*CompatModel)

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		TopK:     intPtr(5),
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range warnings {
		if w.Feature == "topK" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected topK warning from CallWarner, got %+v", warnings)
	}
}

func TestCompatModel_RequestModifier_Warnings(t *testing.T) {
	// RequestModifier warnings should flow through buildRequest too.
	modifier := func(providerOptions map[string]any) (map[string]any, []stream.Warning, error) {
		return nil, []stream.Warning{stream.OtherWarning("something ignored")}, nil
	}

	p := New(Options{
		ProviderID:      "test",
		BaseURL:         "https://example.com",
		APIKey:          "k",
		RequestModifier: modifier,
	})
	m := p.Model("test-model").(*CompatModel)

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range warnings {
		if w.Type == stream.WarningOther && w.Message == "something ignored" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected other warning from RequestModifier, got %+v", warnings)
	}
}
