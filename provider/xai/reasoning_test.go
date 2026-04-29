package xai

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// CallOptions.Reasoning lowers into reasoning.effort on xAI Responses,
// matching the v4 reasoning enum. Provider-specific opts.ReasoningEffort
// takes precedence when both are set.

func captureXaiBody(t *testing.T, callOpts *stream.CallOptions) map[string]any {
	t.Helper()
	var captured map[string]any
	server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
	defer server.Close()

	p := newTestProvider(server.URL)
	m := p.Responses("grok-4")
	events, err := m.Stream(context.Background(), callOpts)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	for range events {
	}
	return captured
}

func TestXaiResponses_ReasoningLowersToWire(t *testing.T) {
	body := captureXaiBody(t, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortMedium,
	})
	r, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("expected reasoning object, got %T (%v)", body["reasoning"], body["reasoning"])
	}
	if r["effort"] != "medium" {
		t.Errorf("reasoning.effort = %v, want medium", r["effort"])
	}
}

func TestXaiResponses_ProviderEffortWinsOverReasoning(t *testing.T) {
	body := captureXaiBody(t, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortLow,
		ProviderOptions: map[string]any{
			"reasoningEffort": "high",
		},
	})
	r, _ := body["reasoning"].(map[string]any)
	if r["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want high (provider option overrides Reasoning)", r["effort"])
	}
}
