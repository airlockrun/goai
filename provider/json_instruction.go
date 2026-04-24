package provider

import (
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/message"
)

// Port of ai-sdk packages/provider-utils/src/inject-json-instruction.ts.
// Used by providers that lack a native structured-output API to coax JSON
// output from the model via a system-message instruction.

const (
	jsonSchemaPrefix   = "JSON schema:"
	jsonSchemaSuffix   = "You MUST answer with a JSON object that matches the JSON schema above."
	jsonGenericSuffix  = "You MUST answer with JSON."
)

// BuildJSONInstruction returns the instruction text that is injected into the
// system message. If prompt is non-empty it is kept as the first line. If
// schema is non-nil, the schema and schema-aware suffix are appended; otherwise
// a generic "answer with JSON" suffix is appended.
func BuildJSONInstruction(prompt string, schema json.RawMessage) string {
	var parts []string
	if prompt != "" {
		parts = append(parts, prompt, "")
	}
	if schema != nil {
		parts = append(parts, jsonSchemaPrefix, string(schema), jsonSchemaSuffix)
	} else {
		parts = append(parts, jsonGenericSuffix)
	}
	return strings.Join(parts, "\n")
}

// InjectJSONInstruction returns a new message slice with JSON-formatting
// instructions merged into (or prepended as) a system message.
//
// Mirrors ai-sdk's injectJsonInstructionIntoMessages. If the first message is
// a system message, its text is augmented; otherwise a new system message is
// prepended. The input slice is not mutated.
func InjectJSONInstruction(messages []message.Message, schema json.RawMessage) []message.Message {
	var existing string
	hasSystem := len(messages) > 0 && messages[0].Role == message.RoleSystem
	if hasSystem {
		existing = messages[0].Content.Text
	}

	injected := message.Message{
		Role:    message.RoleSystem,
		Content: message.Content{Text: BuildJSONInstruction(existing, schema)},
	}
	if hasSystem {
		injected.ProviderOptions = messages[0].ProviderOptions
	}

	out := make([]message.Message, 0, len(messages)+1)
	out = append(out, injected)
	if hasSystem {
		out = append(out, messages[1:]...)
	} else {
		out = append(out, messages...)
	}
	return out
}
