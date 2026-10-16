package schema

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

func TestFromTypeMaps(t *testing.T) {
	type stringKey string
	type entry struct {
		Count int `json:"count"`
	}
	for _, tc := range []struct {
		name    string
		value   any
		valid   string
		invalid string
	}{
		{"string", map[string]string{}, `{"key":"value"}`, `{"key":1}`},
		{"integer", map[string]int{}, `{"key":1}`, `{"key":"1"}`},
		{"named string key", map[stringKey]int{}, `{"key":1}`, `{"key":"1"}`},
		{"fixed parent", struct {
			Values map[string]int `json:"values"`
		}{}, `{"values":{"key":1}}`, `{"values":{"key":1},"extra":2}`},
		{"boolean", map[string]bool{}, `{"key":true}`, `{"key":"true"}`},
		{"number", map[string]float64{}, `{"key":1.5}`, `{"key":"1.5"}`},
		{"struct", map[string]entry{}, `{"key":{"count":1}}`, `{"key":{"count":1,"extra":2}}`},
		{"nested", map[string]map[string]int{}, `{"key":{"nested":1}}`, `{"key":{"nested":"1"}}`},
		{"slice", map[string][]int{}, `{"key":[1,2]}`, `{"key":["1"]}`},
		{"nullable value", map[string]*entry{}, `{"key":null,"other":{"count":1}}`, `{"key":{"count":"1"}}`},
		{"pointer map", new(map[string]int), `null`, `{"key":"1"}`},
		{"slice of maps", []map[string]int{}, `[{"key":1}]`, `[{"key":"1"}]`},
		{"any", map[string]any{}, `{"key":[null,1,"text",{"nested":true}]}`, ``},
	} {
		t.Run(tc.name, func(t *testing.T) {
			generated := MustFromType(tc.value)
			for _, strict := range []bool{false, true} {
				s := generated
				if strict {
					s = ToStrict(s)
				}
				raw := s.MustJSON()
				var standard jsonschema.Schema
				if err := json.Unmarshal(raw, &standard); err != nil {
					t.Fatal(err)
				}
				resolved, err := standard.Resolve(nil)
				if err != nil {
					t.Fatal(err)
				}
				for _, sample := range []struct {
					data string
					ok   bool
				}{{tc.valid, true}, {tc.invalid, false}} {
					if sample.data == "" {
						continue
					}
					var value any
					if err := json.Unmarshal([]byte(sample.data), &value); err != nil {
						t.Fatal(err)
					}
					if err := resolved.Validate(value); (err == nil) != sample.ok {
						t.Fatalf("strict=%v validate %s: %v; schema=%s", strict, sample.data, err, raw)
					}
				}
				var roundTrip Schema
				if err := json.Unmarshal(raw, &roundTrip); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(raw, roundTrip.MustJSON()) {
					t.Fatalf("schema changed during round trip: %s -> %s", raw, roundTrip.MustJSON())
				}
			}
			mapSchema := generated
			if len(mapSchema.AnyOf) > 0 {
				mapSchema = mapSchema.AnyOf[0]
			} else if mapSchema.Type == "array" {
				mapSchema = mapSchema.Items
			}
			if mapSchema.Items != nil {
				t.Fatal("object dictionary must not have items")
			}
			if tc.name == "any" && (mapSchema.AdditionalProperties == nil || !*mapSchema.AdditionalProperties) {
				t.Fatal("map[string]any must have additionalProperties: true")
			}
		})
	}
	for _, value := range []any{map[int]string{}, map[bool]string{}} {
		if _, err := FromType(value); err == nil {
			t.Fatalf("accepted unsupported map keys: %T", value)
		}
	}
}

func TestDictionarySchemaReferences(t *testing.T) {
	s := &Schema{
		Type:                       "object",
		AdditionalPropertiesSchema: &Schema{Ref: "#/$defs/value"},
		Definitions:                map[string]*Schema{"value": {Type: "object", Properties: map[string]*Schema{"count": Integer()}}},
	}
	strict := ToStrict(s)
	if strict.AdditionalPropertiesSchema.Ref != "#/$defs/value" || strict.Definitions["value"].AdditionalProperties == nil || *strict.Definitions["value"].AdditionalProperties {
		t.Fatalf("unexpected dictionary/ref normalization: %s", strict.MustJSON())
	}
	if s.Definitions["value"].AdditionalProperties != nil || reflect.ValueOf(strict.AdditionalPropertiesSchema).Pointer() == reflect.ValueOf(s.AdditionalPropertiesSchema).Pointer() {
		t.Fatal("ToStrict must copy dictionary schemas and definitions")
	}
	allow := true
	s.AdditionalProperties = &allow
	if _, err := s.JSON(); err == nil {
		t.Fatal("accepted conflicting additionalProperties declarations")
	}
}

func TestNumericFieldTags(t *testing.T) {
	type bounded struct {
		Count    int      `json:"count" minimum:"0" maximum:"1000000"`
		Exact    int64    `json:"exact" minimum:"-9223372036854775808" maximum:"9223372036854775807"`
		Fraction *float64 `json:"fraction" minimum:"0.125" maximum:"1e100"`
	}
	raw := MustFromType(bounded{}).MustJSON()
	for _, expected := range []string{`"minimum":0`, `"maximum":1000000`, `"minimum":-9223372036854775808`, `"maximum":9223372036854775807`, `"minimum":0.125`, `"maximum":1e100`} {
		if !bytes.Contains(raw, []byte(expected)) {
			t.Fatalf("missing %s in %s", expected, raw)
		}
	}
	for _, value := range []any{
		struct {
			Count int `minimum:"invalid"`
		}{},
		struct {
			Count int `maximum:"\"100\""`
		}{},
		struct {
			Count int `minimum:"10" maximum:"1"`
		}{},
		struct {
			Text string `minimum:"0"`
		}{},
	} {
		if _, err := FromType(value); err == nil {
			t.Fatalf("accepted invalid tags on %T", value)
		}
	}
}

func TestGeneratedSchemasDoNotInventValidationLimits(t *testing.T) {
	type unannotated struct {
		Text     string            `json:"text"`
		Count    int               `json:"count"`
		Unsigned uint64            `json:"unsigned"`
		Fraction float64           `json:"fraction"`
		Enabled  bool              `json:"enabled"`
		Values   []string          `json:"values"`
		Fixed    [2]int            `json:"fixed"`
		Labels   map[string]string `json:"labels"`
		Optional *int              `json:"optional,omitempty"`
	}
	raw := MustFromType(unannotated{}).MustJSON()
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatal(err)
	}
	var inspect func(any)
	inspect = func(value any) {
		switch v := value.(type) {
		case map[string]any:
			for _, keyword := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern", "format", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties", "default", "enum"} {
				if _, ok := v[keyword]; ok {
					t.Fatalf("invented %s constraint: %s", keyword, raw)
				}
			}
			for _, child := range v {
				inspect(child)
			}
		case []any:
			for _, child := range v {
				inspect(child)
			}
		}
	}
	inspect(root)
	properties := root["properties"].(map[string]any)
	for name, kind := range map[string]string{"text": "string", "count": "integer", "unsigned": "integer", "fraction": "number", "enabled": "boolean"} {
		property := properties[name].(map[string]any)
		if len(property) != 1 || property["type"] != kind {
			t.Fatalf("unexpected %s schema: %s", name, raw)
		}
	}
}

// Tests for ToStrict (and schema utilities) - translated from ai-sdk
// Source: ai-sdk/packages/provider-utils/src/add-additional-properties-to-json-schema.test.ts

func TestToStrict_RecursiveObjects(t *testing.T) {
	t.Run("adds additionalProperties false to objects recursively", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"user": {
					Type: "object",
					Properties: map[string]*Schema{
						"name": {Type: "string"},
					},
				},
				"age": {Type: "number"},
			},
		}

		result := ToStrict(schema)

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be false")
		}

		userSchema := result.Properties["user"]
		if *userSchema.AdditionalProperties != false {
			t.Error("expected nested user additionalProperties to be false")
		}
	})
}

func TestToStrict_ArraysWithObjects(t *testing.T) {
	t.Run("adds additionalProperties false to objects inside arrays", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"ingredients": {
					Type: "array",
					Items: &Schema{
						Type: "object",
						Properties: map[string]*Schema{
							"name":   {Type: "string"},
							"amount": {Type: "string"},
						},
						Required: []string{"name", "amount"},
					},
				},
			},
			Required: []string{"ingredients"},
		}

		result := ToStrict(schema)

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be false")
		}

		ingredientsItems := result.Properties["ingredients"].Items
		if *ingredientsItems.AdditionalProperties != false {
			t.Error("expected array items additionalProperties to be false")
		}
	})
}

func TestToStrict_AnyOf(t *testing.T) {
	t.Run("adds additionalProperties false to objects inside anyOf", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"response": {
					AnyOf: []*Schema{
						{Type: "object", Properties: map[string]*Schema{"name": {Type: "string"}}},
						{Type: "object", Properties: map[string]*Schema{"amount": {Type: "string"}}},
					},
				},
			},
		}

		result := ToStrict(schema)

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be false")
		}

		responseSchema := result.Properties["response"]
		for i, sub := range responseSchema.AnyOf {
			if sub.Type == "object" && sub.AdditionalProperties != nil && *sub.AdditionalProperties != false {
				t.Errorf("expected anyOf[%d] additionalProperties to be false", i)
			}
		}
	})
}

func TestToStrict_AllOf(t *testing.T) {
	t.Run("adds additionalProperties false to objects inside allOf", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"response": {
					AllOf: []*Schema{
						{Type: "object", Properties: map[string]*Schema{"name": {Type: "string"}}},
						{Type: "object", Properties: map[string]*Schema{"age": {Type: "number"}}},
					},
				},
			},
		}

		result := ToStrict(schema)

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be false")
		}

		responseSchema := result.Properties["response"]
		for i, sub := range responseSchema.AllOf {
			if sub.Type == "object" && sub.AdditionalProperties != nil && *sub.AdditionalProperties != false {
				t.Errorf("expected allOf[%d] additionalProperties to be false", i)
			}
		}
	})
}

func TestToStrict_OneOf(t *testing.T) {
	t.Run("adds additionalProperties false to objects inside oneOf", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"response": {
					OneOf: []*Schema{
						{Type: "object", Properties: map[string]*Schema{"success": {Type: "boolean"}}},
						{Type: "object", Properties: map[string]*Schema{"error": {Type: "string"}}},
					},
				},
			},
		}

		result := ToStrict(schema)

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be false")
		}

		responseSchema := result.Properties["response"]
		for i, sub := range responseSchema.OneOf {
			if sub.Type == "object" && sub.AdditionalProperties != nil && *sub.AdditionalProperties != false {
				t.Errorf("expected oneOf[%d] additionalProperties to be false", i)
			}
		}
	})
}

func TestToStrict_PreservesExplicitDictionaries(t *testing.T) {
	t.Run("preserves existing additionalProperties flags", func(t *testing.T) {
		additionalTrue := true
		schema := &Schema{
			Type:                 "object",
			AdditionalProperties: &additionalTrue,
			Properties: map[string]*Schema{
				"meta": {
					Type:                 "object",
					AdditionalProperties: &additionalTrue,
					Properties: map[string]*Schema{
						"id": {Type: "string"},
					},
				},
			},
		}

		result := ToStrict(schema)

		if !*result.AdditionalProperties {
			t.Error("expected root additionalProperties to remain true")
		}

		metaSchema := result.Properties["meta"]
		if !*metaSchema.AdditionalProperties {
			t.Error("expected nested additionalProperties to remain true")
		}
	})
}

func TestToStrict_NonObjectSchemas(t *testing.T) {
	t.Run("leaves non-object schemas unchanged", func(t *testing.T) {
		schema := &Schema{Type: "string"}

		result := ToStrict(schema)

		if result.Type != "string" {
			t.Errorf("expected type 'string', got '%s'", result.Type)
		}
		if result.AdditionalProperties != nil {
			t.Error("expected additionalProperties to be nil for non-object schema")
		}
	})

	t.Run("leaves number schema unchanged", func(t *testing.T) {
		schema := &Schema{Type: "number"}

		result := ToStrict(schema)

		if result.Type != "number" {
			t.Errorf("expected type 'number', got '%s'", result.Type)
		}
	})

	t.Run("leaves array of primitives unchanged", func(t *testing.T) {
		schema := &Schema{
			Type:  "array",
			Items: &Schema{Type: "string"},
		}

		result := ToStrict(schema)

		if result.Type != "array" {
			t.Errorf("expected type 'array', got '%s'", result.Type)
		}
		if result.Items.Type != "string" {
			t.Errorf("expected items type 'string', got '%s'", result.Items.Type)
		}
	})
}

func TestToStrict_RequiredProperties(t *testing.T) {
	t.Run("ensures all properties are in required array", func(t *testing.T) {
		schema := &Schema{
			Type: "object",
			Properties: map[string]*Schema{
				"name":  {Type: "string"},
				"age":   {Type: "number"},
				"email": {Type: "string"},
			},
		}

		result := ToStrict(schema)

		if len(result.Required) != 3 {
			t.Fatalf("expected 3 required properties, got %d", len(result.Required))
		}

		requiredSet := make(map[string]bool)
		for _, r := range result.Required {
			requiredSet[r] = true
		}

		if !requiredSet["name"] {
			t.Error("expected 'name' to be in required")
		}
		if !requiredSet["age"] {
			t.Error("expected 'age' to be in required")
		}
		if !requiredSet["email"] {
			t.Error("expected 'email' to be in required")
		}
	})
}

// Test schema builder functions

func TestSchemaBuilders(t *testing.T) {
	t.Run("Object creates schema with additionalProperties false", func(t *testing.T) {
		schema := Object(map[string]*Schema{
			"name": String(),
			"age":  Integer(),
		})

		if schema.Type != "object" {
			t.Errorf("expected type 'object', got '%s'", schema.Type)
		}
		if *schema.AdditionalProperties != false {
			t.Error("expected additionalProperties to be false")
		}
		if len(schema.Properties) != 2 {
			t.Errorf("expected 2 properties, got %d", len(schema.Properties))
		}
		if len(schema.Required) != 2 {
			t.Errorf("expected 2 required, got %d", len(schema.Required))
		}
	})

	t.Run("Array creates array schema", func(t *testing.T) {
		schema := Array(String())

		if schema.Type != "array" {
			t.Errorf("expected type 'array', got '%s'", schema.Type)
		}
		if schema.Items == nil {
			t.Error("expected items to be set")
		}
		if schema.Items.Type != "string" {
			t.Errorf("expected items type 'string', got '%s'", schema.Items.Type)
		}
	})

	t.Run("String creates string schema", func(t *testing.T) {
		schema := String()
		if schema.Type != "string" {
			t.Errorf("expected type 'string', got '%s'", schema.Type)
		}
	})

	t.Run("Number creates number schema", func(t *testing.T) {
		schema := Number()
		if schema.Type != "number" {
			t.Errorf("expected type 'number', got '%s'", schema.Type)
		}
	})

	t.Run("Integer creates integer schema", func(t *testing.T) {
		schema := Integer()
		if schema.Type != "integer" {
			t.Errorf("expected type 'integer', got '%s'", schema.Type)
		}
	})

	t.Run("Boolean creates boolean schema", func(t *testing.T) {
		schema := Boolean()
		if schema.Type != "boolean" {
			t.Errorf("expected type 'boolean', got '%s'", schema.Type)
		}
	})

	t.Run("StringEnum creates string enum schema", func(t *testing.T) {
		schema := StringEnum("foo", "bar", "baz")
		if schema.Type != "string" {
			t.Errorf("expected type 'string', got '%s'", schema.Type)
		}
		if len(schema.Enum) != 3 {
			t.Errorf("expected 3 enum values, got %d", len(schema.Enum))
		}
	})
}

func TestFromType(t *testing.T) {
	t.Run("creates schema from struct type", func(t *testing.T) {
		type Person struct {
			Name  string `json:"name"`
			Age   int    `json:"age"`
			Email string `json:"email,omitempty"`
		}

		schema, err := FromType(Person{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if schema.Type != "object" {
			t.Errorf("expected type 'object', got '%s'", schema.Type)
		}
		if len(schema.Properties) != 3 {
			t.Errorf("expected 3 properties, got %d", len(schema.Properties))
		}
		if schema.Properties["name"] == nil {
			t.Error("expected 'name' property")
		}
		if schema.Properties["age"] == nil {
			t.Error("expected 'age' property")
		} else if schema.Properties["age"].Type != "integer" {
			t.Errorf("age type = %q, want integer", schema.Properties["age"].Type)
		}
		if schema.Properties["email"] == nil {
			t.Error("expected 'email' property")
		}

		// Check additionalProperties is false (strict mode by default)
		if *schema.AdditionalProperties != false {
			t.Error("expected additionalProperties to be false")
		}
	})

	t.Run("handles nested structs", func(t *testing.T) {
		type Address struct {
			Street string `json:"street"`
			City   string `json:"city"`
		}
		type Person struct {
			Name    string  `json:"name"`
			Address Address `json:"address"`
		}

		schema, err := FromType(Person{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		addressSchema := schema.Properties["address"]
		if addressSchema == nil {
			t.Fatal("expected 'address' property")
		}
		if addressSchema.Type != "object" {
			t.Errorf("expected address type 'object', got '%s'", addressSchema.Type)
		}
		if addressSchema.Properties["street"] == nil {
			t.Error("expected 'street' property in address")
		}
	})
}

func TestSchemaMarshalJSON(t *testing.T) {
	t.Run("object schema always has properties field", func(t *testing.T) {
		schema := &Schema{Type: "object"}

		data, err := json.Marshal(schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if _, ok := result["properties"]; !ok {
			t.Error("expected 'properties' field in marshaled object schema")
		}
	})

	t.Run("non-object schema does not have properties field when empty", func(t *testing.T) {
		schema := &Schema{Type: "string"}

		data, err := json.Marshal(schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var result map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if _, ok := result["properties"]; ok {
			t.Error("expected no 'properties' field in marshaled string schema")
		}
	})
}
