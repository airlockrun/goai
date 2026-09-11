package xai

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func TestStableReasoningPolicy(t *testing.T) {
	for _, tc := range []struct {
		id, effort, want string
		warning          bool
	}{
		{"grok-4.5", "xhigh", "high", true}, {"grok-4.6", "xhigh", "xhigh", false}, {"grok-4.20-reasoning", "high", "", true}, {"grok-4.20-0309-non-reasoning", "none", "", true}, {"grok-4.20-multi-agent", "minimal", "low", true}, {"grok-4.6", "provider-default", "", false},
	} {
		t.Run(tc.id+tc.effort, func(t *testing.T) {
			m := New(Options{}).Responses(tc.id).(*XaiResponsesModel)
			data, warnings, err := m.buildRequest(&stream.CallOptions{Reasoning: tc.effort})
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Reasoning *struct {
					Effort string `json:"effort"`
				} `json:"reasoning"`
			}
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			got := ""
			if body.Reasoning != nil {
				got = body.Reasoning.Effort
			}
			if got != tc.want || (len(warnings) > 0) != tc.warning {
				t.Fatalf("effort=%q warnings=%v", got, warnings)
			}
		})
	}
}

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
