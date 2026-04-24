package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
)

func TestBuildJSONInstruction(t *testing.T) {
	tests := []struct {
		name     string
		prompt   string
		schema   json.RawMessage
		wantHas  []string
		wantSkip []string
	}{
		{
			name:    "empty prompt with schema",
			prompt:  "",
			schema:  json.RawMessage(`{"type":"object"}`),
			wantHas: []string{jsonSchemaPrefix, `{"type":"object"}`, jsonSchemaSuffix},
		},
		{
			name:     "empty prompt without schema",
			prompt:   "",
			schema:   nil,
			wantHas:  []string{jsonGenericSuffix},
			wantSkip: []string{jsonSchemaPrefix, jsonSchemaSuffix},
		},
		{
			name:    "existing prompt with schema",
			prompt:  "You are a helpful assistant.",
			schema:  json.RawMessage(`{"type":"object"}`),
			wantHas: []string{"You are a helpful assistant.", jsonSchemaPrefix, jsonSchemaSuffix},
		},
		{
			name:     "existing prompt without schema",
			prompt:   "You are a helpful assistant.",
			schema:   nil,
			wantHas:  []string{"You are a helpful assistant.", jsonGenericSuffix},
			wantSkip: []string{jsonSchemaPrefix},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildJSONInstruction(tt.prompt, tt.schema)
			for _, s := range tt.wantHas {
				if !strings.Contains(got, s) {
					t.Errorf("expected instruction to contain %q, got %q", s, got)
				}
			}
			for _, s := range tt.wantSkip {
				if strings.Contains(got, s) {
					t.Errorf("expected instruction NOT to contain %q, got %q", s, got)
				}
			}
		})
	}
}

func TestInjectJSONInstruction_PrependsWhenNoSystemMessage(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("hello"),
	}
	out := InjectJSONInstruction(msgs, nil)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	if out[0].Role != message.RoleSystem {
		t.Errorf("expected first message to be system, got %s", out[0].Role)
	}
	if !strings.Contains(out[0].Content.Text, jsonGenericSuffix) {
		t.Errorf("expected system message to contain generic suffix, got %q", out[0].Content.Text)
	}
	if out[1].Role != message.RoleUser || out[1].Content.Text != "hello" {
		t.Errorf("user message should be preserved, got %+v", out[1])
	}
}

func TestInjectJSONInstruction_AugmentsExistingSystemMessage(t *testing.T) {
	msgs := []message.Message{
		message.NewSystemMessage("You are a helpful assistant."),
		message.NewUserMessage("hello"),
	}
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
	out := InjectJSONInstruction(msgs, schema)

	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	sys := out[0]
	if sys.Role != message.RoleSystem {
		t.Errorf("expected first message to be system, got %s", sys.Role)
	}
	if !strings.Contains(sys.Content.Text, "You are a helpful assistant.") {
		t.Errorf("existing system text should be preserved, got %q", sys.Content.Text)
	}
	if !strings.Contains(sys.Content.Text, jsonSchemaPrefix) {
		t.Errorf("expected schema prefix, got %q", sys.Content.Text)
	}
	if !strings.Contains(sys.Content.Text, string(schema)) {
		t.Errorf("expected schema JSON, got %q", sys.Content.Text)
	}
}

func TestInjectJSONInstruction_DoesNotMutateInput(t *testing.T) {
	original := []message.Message{
		message.NewSystemMessage("original"),
		message.NewUserMessage("hi"),
	}
	_ = InjectJSONInstruction(original, nil)
	if original[0].Content.Text != "original" {
		t.Errorf("input was mutated: system text became %q", original[0].Content.Text)
	}
	if len(original) != 2 {
		t.Errorf("input length was mutated: %d", len(original))
	}
}
