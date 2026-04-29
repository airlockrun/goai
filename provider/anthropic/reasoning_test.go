package anthropic

import (
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// CallOptions.Reasoning is the uniform v4 effort enum. It lowers into
// output_config.effort the same way provider-specific MessagesOptions.Effort
// does, but provider-specific Effort wins when both are set.

func TestReasoning_LowersToOutputConfigEffort(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortMedium,
	})

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected output_config object, got %T (%v)", body["output_config"], body["output_config"])
	}
	if oc["effort"] != "medium" {
		t.Errorf("output_config.effort = %v, want medium", oc["effort"])
	}
}

// Provider-specific MessagesOptions.Effort takes precedence over the
// top-level Reasoning enum so callers can keep an explicit override.
func TestReasoning_ProviderEffortWinsOverTopLevel(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortLow,
		ProviderOptions: map[string]any{
			"effort": "high",
		},
	})

	oc, _ := body["output_config"].(map[string]any)
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want high (provider effort overrides Reasoning)", oc["effort"])
	}
}

// Reasoning is suppressed when Thinking.Type=="disabled" (same gate as
// the existing Effort suppression).
func TestReasoning_SuppressedWhenThinkingDisabled(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortHigh,
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "disabled"},
		},
	})
	if _, has := body["output_config"]; has {
		t.Errorf("expected no output_config when thinking.type=disabled, got %v", body["output_config"])
	}
}

func TestReasoning_OmittedWhenUnset(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if _, has := body["output_config"]; has {
		t.Errorf("expected no output_config when Reasoning is unset, got %v", body["output_config"])
	}
}
