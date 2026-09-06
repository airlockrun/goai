package cohere

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int { return &v }

func TestCohere_BuildRequest_Seed(t *testing.T) {
	p := New(Options{APIKey: "k"})
	m := p.Model("command-r").(*CohereModel)

	body, warnings, err := m.buildRequest(&stream.CallOptions{
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
	if found {
		t.Errorf("unexpected seed warning: %+v", warnings)
	}
	var request map[string]any
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	if request["seed"] != float64(42) {
		t.Fatalf("seed = %v", request["seed"])
	}
}
