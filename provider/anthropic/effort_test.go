package anthropic

import (
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// effort lives at output_config.effort. Mirrors ai-sdk's
// anthropic-language-model.ts:454-456. Verified via the public
// BuildRequestBody surface so renames inside the assembly logic are caught.

func TestEffort_OnWire(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: map[string]any{
			"effort": "medium",
		},
	})

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected output_config object on wire, got %T (%v)", body["output_config"], body["output_config"])
	}
	if oc["effort"] != "medium" {
		t.Errorf("output_config.effort = %v, want medium", oc["effort"])
	}
}

func TestEffort_OmittedWhenUnset(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if _, has := body["output_config"]; has {
		t.Errorf("expected no output_config when effort is unset, got %v", body["output_config"])
	}
}

// ai-sdk parity: when thinking.type is explicitly "disabled" the model can't
// apply effort to reasoning, so effort is suppressed
// (anthropic-language-model.ts:407-411). Adaptive/enabled thinking lets it
// through.
func TestEffort_SuppressedWhenThinkingDisabled(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: map[string]any{
			"effort":   "high",
			"thinking": map[string]any{"type": "disabled"},
		},
	})
	if _, has := body["output_config"]; has {
		t.Errorf("expected no output_config when thinking.type=disabled, got %v", body["output_config"])
	}
}

func TestEffort_AllowedWithAdaptiveThinking(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: map[string]any{
			"effort":   "high",
			"thinking": map[string]any{"type": "adaptive"},
		},
	})
	oc, _ := body["output_config"].(map[string]any)
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want high", oc["effort"])
	}
	thinking, _ := body["thinking"].(map[string]any)
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
	}
}

// effort + structured-output coexist on the same output_config object —
// neither path should clobber the other's fields.
func TestEffort_CoexistsWithStructuredOutput(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		ResponseFormat: &stream.ResponseFormat{
			Type:   "json",
			Name:   "answer",
			Schema: []byte(`{"type":"object"}`),
		},
		ProviderOptions: map[string]any{
			"effort":               "medium",
			"structuredOutputMode": "outputFormat",
		},
	})

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("expected output_config, got %v", body["output_config"])
	}
	if oc["effort"] != "medium" {
		t.Errorf("output_config.effort = %v, want medium", oc["effort"])
	}
	format, ok := oc["format"].(map[string]any)
	if !ok {
		t.Fatalf("expected output_config.format, got %v", oc["format"])
	}
	wantFormat := map[string]any{
		"type":   "json_schema",
		"name":   "answer",
		"schema": map[string]any{"type": "object"},
	}
	if !reflect.DeepEqual(format, wantFormat) {
		t.Errorf("output_config.format = %#v, want %#v", format, wantFormat)
	}
}
