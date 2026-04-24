package cohere

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int { return &v }

func TestCohere_BuildRequest_SeedWarning(t *testing.T) {
	p := New(Options{APIKey: "k"})
	m := p.Model("command-r").(*CohereModel)

	_, warnings, err := m.buildRequest(&stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		Seed:     intPtr(42),
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range warnings {
		if w.Feature == "seed" && w.Type == stream.WarningUnsupported {
			found = true
		}
	}
	if !found {
		t.Errorf("expected seed warning, got %+v", warnings)
	}
}
