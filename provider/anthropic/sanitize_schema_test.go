package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSanitizeJSONSchema_StripsConstraintsIntoDescription(t *testing.T) {
	in := json.RawMessage(`{
		"type": "object",
		"properties": {
			"age": {"type": "integer", "minimum": 0, "maximum": 120},
			"name": {"type": "string", "minLength": 1, "maxLength": 50, "pattern": "^[a-z]+$"}
		},
		"required": ["age", "name"]
	}`)

	out, err := sanitizeJSONSchema(in)
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	props := got["properties"].(map[string]any)

	age := props["age"].(map[string]any)
	if _, ok := age["minimum"]; ok {
		t.Errorf("expected minimum stripped: %v", age)
	}
	desc, _ := age["description"].(string)
	if !strings.Contains(desc, "minimum: 0") || !strings.Contains(desc, "maximum: 120") {
		t.Errorf("age description = %q", desc)
	}

	name := props["name"].(map[string]any)
	if _, ok := name["pattern"]; ok {
		t.Errorf("expected pattern stripped: %v", name)
	}
	nameDesc, _ := name["description"].(string)
	if !strings.Contains(nameDesc, "min length: 1") || !strings.Contains(nameDesc, "pattern: ^[a-z]+$") {
		t.Errorf("name description = %q", nameDesc)
	}
}

func TestSanitizeJSONSchema_ForcesAdditionalPropertiesFalse(t *testing.T) {
	in := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	out, _ := sanitizeJSONSchema(in)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["additionalProperties"] != false {
		t.Errorf("expected additionalProperties=false, got %v", got["additionalProperties"])
	}
}

func TestSanitizeJSONSchema_PreservesSupportedFormat(t *testing.T) {
	in := json.RawMessage(`{"type":"string","format":"email"}`)
	out, _ := sanitizeJSONSchema(in)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["format"] != "email" {
		t.Errorf("expected format=email preserved, got %v", got["format"])
	}
}

func TestSanitizeJSONSchema_DropsUnsupportedFormatToDescription(t *testing.T) {
	in := json.RawMessage(`{"type":"string","format":"semver"}`)
	out, _ := sanitizeJSONSchema(in)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if _, ok := got["format"]; ok {
		t.Errorf("expected unsupported format to be dropped from output: %v", got)
	}
	desc, _ := got["description"].(string)
	if !strings.Contains(desc, "format: semver") {
		t.Errorf("expected format hint in description, got %q", desc)
	}
}

func TestSanitizeJSONSchema_RefPassthrough(t *testing.T) {
	in := json.RawMessage(`{"$ref":"#/definitions/x","minimum":0,"description":"ignored"}`)
	out, _ := sanitizeJSONSchema(in)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["$ref"] != "#/definitions/x" {
		t.Errorf("expected $ref preserved")
	}
	if _, ok := got["minimum"]; ok {
		t.Errorf("expected siblings of $ref to be dropped")
	}
	if _, ok := got["description"]; ok {
		t.Errorf("expected siblings of $ref to be dropped")
	}
}

func TestSanitizeJSONSchema_OneOfBecomesAnyOf(t *testing.T) {
	in := json.RawMessage(`{"oneOf":[{"type":"string"},{"type":"number"}]}`)
	out, _ := sanitizeJSONSchema(in)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if _, ok := got["oneOf"]; ok {
		t.Errorf("expected oneOf collapsed to anyOf")
	}
	if _, ok := got["anyOf"].([]any); !ok {
		t.Errorf("expected anyOf array, got %v", got["anyOf"])
	}
}
