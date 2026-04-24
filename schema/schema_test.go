package schema

import (
	"encoding/json"
	"testing"
)

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

func TestToStrict_OverwritesExisting(t *testing.T) {
	t.Run("overwrites existing additionalProperties flags", func(t *testing.T) {
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

		if *result.AdditionalProperties != false {
			t.Error("expected root additionalProperties to be overwritten to false")
		}

		metaSchema := result.Properties["meta"]
		if *metaSchema.AdditionalProperties != false {
			t.Error("expected nested additionalProperties to be overwritten to false")
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
