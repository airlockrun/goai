package google

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Ported from ai-sdk packages/google/src/convert-json-schema-to-openapi-schema.test.ts.

func TestConvertJSONSchemaToOpenAPI(t *testing.T) {
	parse := func(t *testing.T, raw json.RawMessage) map[string]any {
		t.Helper()
		if raw == nil {
			return nil
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}

	t.Run("empty object schema at root returns nil", func(t *testing.T) {
		got := convertJSONSchemaToOpenAPI(json.RawMessage(`{"type":"object"}`))
		if got != nil {
			t.Errorf("expected nil, got %s", string(got))
		}
	})

	t.Run("nested empty object preserved", func(t *testing.T) {
		raw := json.RawMessage(`{"type":"object","properties":{"inner":{"type":"object"}}}`)
		got := parse(t, convertJSONSchemaToOpenAPI(raw))
		props, _ := got["properties"].(map[string]any)
		inner, _ := props["inner"].(map[string]any)
		if inner["type"] != "object" {
			t.Errorf("expected nested object preserved, got %v", inner)
		}
	})

	t.Run("nullable via type array collapses to nullable+anyOf", func(t *testing.T) {
		// type: ["string", "null"] → anyOf:[{type:"string"}], nullable:true
		raw := json.RawMessage(`{"type":"object","properties":{"x":{"type":["string","null"]}}}`)
		got := parse(t, convertJSONSchemaToOpenAPI(raw))
		x := got["properties"].(map[string]any)["x"].(map[string]any)
		if x["nullable"] != true {
			t.Errorf("expected nullable=true, got %v", x)
		}
		anyOf, _ := x["anyOf"].([]any)
		if len(anyOf) != 1 {
			t.Fatalf("expected anyOf length 1, got %v", anyOf)
		}
		first, _ := anyOf[0].(map[string]any)
		if first["type"] != "string" {
			t.Errorf("expected anyOf[0].type=string, got %v", first)
		}
	})

	t.Run("strips $schema", func(t *testing.T) {
		raw := json.RawMessage(`{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"x":{"type":"string"}}}`)
		got := parse(t, convertJSONSchemaToOpenAPI(raw))
		if _, has := got["$schema"]; has {
			t.Errorf("expected $schema stripped, got %v", got)
		}
	})

	t.Run("const maps to single-element enum", func(t *testing.T) {
		raw := json.RawMessage(`{"type":"object","properties":{"mode":{"const":"a"}}}`)
		got := parse(t, convertJSONSchemaToOpenAPI(raw))
		mode := got["properties"].(map[string]any)["mode"].(map[string]any)
		enum, _ := mode["enum"].([]any)
		if !reflect.DeepEqual(enum, []any{"a"}) {
			t.Errorf("expected enum [\"a\"], got %v", enum)
		}
	})

	t.Run("anyOf with null schema becomes nullable", func(t *testing.T) {
		raw := json.RawMessage(`{"type":"object","properties":{"x":{"anyOf":[{"type":"string"},{"type":"null"}]}}}`)
		got := parse(t, convertJSONSchemaToOpenAPI(raw))
		x := got["properties"].(map[string]any)["x"].(map[string]any)
		if x["nullable"] != true {
			t.Errorf("expected nullable=true, got %v", x)
		}
		if x["type"] != "string" {
			t.Errorf("expected merged type=string, got %v", x)
		}
	})
}
