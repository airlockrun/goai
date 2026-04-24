// Package schema provides JSON Schema utilities for structured output.
package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
)

// Schema represents a JSON Schema.
type Schema struct {
	// SchemaURI is the $schema URI (e.g., "https://json-schema.org/draft/2020-12/schema").
	SchemaURI string `json:"$schema,omitempty"`

	// Type is the JSON type (object, array, string, number, integer, boolean, null).
	Type string `json:"type,omitempty"`

	// Title is a short description of the schema.
	Title string `json:"title,omitempty"`

	// Description is a detailed description of the schema.
	Description string `json:"description,omitempty"`

	// Properties are the object properties (for object types).
	Properties map[string]*Schema `json:"properties,omitempty"`

	// Required lists required property names.
	Required []string `json:"required,omitempty"`

	// AdditionalProperties controls additional properties (for object types).
	AdditionalProperties *bool `json:"additionalProperties,omitempty"`

	// Items is the schema for array items (for array types).
	Items *Schema `json:"items,omitempty"`

	// Enum lists allowed values.
	Enum []any `json:"enum,omitempty"`

	// Const is the only allowed value.
	Const any `json:"const,omitempty"`

	// Default is the default value.
	Default any `json:"default,omitempty"`

	// MinLength is the minimum string length.
	MinLength *int `json:"minLength,omitempty"`

	// MaxLength is the maximum string length.
	MaxLength *int `json:"maxLength,omitempty"`

	// Pattern is a regex pattern for strings.
	Pattern string `json:"pattern,omitempty"`

	// Format is a semantic format (e.g., "email", "date-time").
	Format string `json:"format,omitempty"`

	// Minimum is the minimum numeric value.
	Minimum *float64 `json:"minimum,omitempty"`

	// Maximum is the maximum numeric value.
	Maximum *float64 `json:"maximum,omitempty"`

	// ExclusiveMinimum is the exclusive minimum numeric value.
	ExclusiveMinimum *float64 `json:"exclusiveMinimum,omitempty"`

	// ExclusiveMaximum is the exclusive maximum numeric value.
	ExclusiveMaximum *float64 `json:"exclusiveMaximum,omitempty"`

	// MultipleOf specifies that the value must be a multiple of this number.
	MultipleOf *float64 `json:"multipleOf,omitempty"`

	// MinItems is the minimum array length.
	MinItems *int `json:"minItems,omitempty"`

	// MaxItems is the maximum array length.
	MaxItems *int `json:"maxItems,omitempty"`

	// UniqueItems requires array items to be unique.
	UniqueItems *bool `json:"uniqueItems,omitempty"`

	// AnyOf allows the value to match any of the sub-schemas.
	AnyOf []*Schema `json:"anyOf,omitempty"`

	// OneOf requires the value to match exactly one sub-schema.
	OneOf []*Schema `json:"oneOf,omitempty"`

	// AllOf requires the value to match all sub-schemas.
	AllOf []*Schema `json:"allOf,omitempty"`

	// Not requires the value to not match the sub-schema.
	Not *Schema `json:"not,omitempty"`

	// Ref is a reference to another schema.
	Ref string `json:"$ref,omitempty"`

	// CustomRef is a non-standard reference used by some tools (e.g., opencode).
	// This is "ref" (not "$ref") and typically contains a type name.
	CustomRef string `json:"-"` // Handled in MarshalJSON

	// Definitions contains reusable schema definitions.
	Definitions map[string]*Schema `json:"$defs,omitempty"`
}

// MarshalJSON implements custom JSON marshaling to ensure object schemas
// always have a "properties" field (even if empty), as required by OpenAI.
// Uses ordered map to ensure $schema comes first.
func (s *Schema) MarshalJSON() ([]byte, error) {
	m := make(map[string]any)

	// $schema should come first (handled by ordered output below)
	if s.SchemaURI != "" {
		m["$schema"] = s.SchemaURI
	}
	if s.Type != "" {
		m["type"] = s.Type
	}
	if s.Title != "" {
		m["title"] = s.Title
	}
	if s.Description != "" {
		m["description"] = s.Description
	}

	// For object types, always include properties (even if empty)
	if s.Type == "object" {
		if s.Properties == nil {
			m["properties"] = map[string]*Schema{}
		} else {
			m["properties"] = s.Properties
		}
	} else if s.Properties != nil && len(s.Properties) > 0 {
		m["properties"] = s.Properties
	}

	if len(s.Required) > 0 {
		m["required"] = s.Required
	}
	if s.AdditionalProperties != nil {
		m["additionalProperties"] = *s.AdditionalProperties
	}
	if s.Items != nil {
		m["items"] = s.Items
	}
	if len(s.Enum) > 0 {
		m["enum"] = s.Enum
	}
	if s.Const != nil {
		m["const"] = s.Const
	}
	if s.Default != nil {
		m["default"] = s.Default
	}
	if s.MinLength != nil {
		m["minLength"] = *s.MinLength
	}
	if s.MaxLength != nil {
		m["maxLength"] = *s.MaxLength
	}
	if s.Pattern != "" {
		m["pattern"] = s.Pattern
	}
	if s.Format != "" {
		m["format"] = s.Format
	}
	if s.Minimum != nil {
		m["minimum"] = *s.Minimum
	}
	if s.Maximum != nil {
		m["maximum"] = *s.Maximum
	}
	if s.ExclusiveMinimum != nil {
		m["exclusiveMinimum"] = *s.ExclusiveMinimum
	}
	if s.ExclusiveMaximum != nil {
		m["exclusiveMaximum"] = *s.ExclusiveMaximum
	}
	if s.MultipleOf != nil {
		m["multipleOf"] = *s.MultipleOf
	}
	if s.MinItems != nil {
		m["minItems"] = *s.MinItems
	}
	if s.MaxItems != nil {
		m["maxItems"] = *s.MaxItems
	}
	if s.UniqueItems != nil {
		m["uniqueItems"] = *s.UniqueItems
	}
	if len(s.AnyOf) > 0 {
		m["anyOf"] = s.AnyOf
	}
	if len(s.OneOf) > 0 {
		m["oneOf"] = s.OneOf
	}
	if len(s.AllOf) > 0 {
		m["allOf"] = s.AllOf
	}
	if s.Not != nil {
		m["not"] = s.Not
	}
	if s.Ref != "" {
		m["$ref"] = s.Ref
	}
	if s.CustomRef != "" {
		m["ref"] = s.CustomRef
	}
	if len(s.Definitions) > 0 {
		m["$defs"] = s.Definitions
	}

	return json.Marshal(m)
}

// JSON returns the schema as a JSON-encoded byte slice.
func (s *Schema) JSON() ([]byte, error) {
	return json.Marshal(s)
}

// MustJSON returns the schema as JSON or panics.
func (s *Schema) MustJSON() []byte {
	b, err := s.JSON()
	if err != nil {
		panic(err)
	}
	return b
}

// String returns the schema as a JSON string.
func (s *Schema) String() string {
	b, _ := json.MarshalIndent(s, "", "  ")
	return string(b)
}

// Object creates an object schema.
func Object(properties map[string]*Schema) *Schema {
	// Sort property names for deterministic required array (important for prompt caching)
	required := make([]string, 0, len(properties))
	for name := range properties {
		required = append(required, name)
	}
	sort.Strings(required)
	additionalProperties := false
	return &Schema{
		Type:                 "object",
		Properties:           properties,
		Required:             required,
		AdditionalProperties: &additionalProperties,
	}
}

// Array creates an array schema.
func Array(items *Schema) *Schema {
	return &Schema{
		Type:  "array",
		Items: items,
	}
}

// String creates a string schema.
func String() *Schema {
	return &Schema{Type: "string"}
}

// StringEnum creates a string schema with allowed values.
func StringEnum(values ...string) *Schema {
	enum := make([]any, len(values))
	for i, v := range values {
		enum[i] = v
	}
	return &Schema{Type: "string", Enum: enum}
}

// Number creates a number schema.
func Number() *Schema {
	return &Schema{Type: "number"}
}

// Integer creates an integer schema.
func Integer() *Schema {
	return &Schema{Type: "integer"}
}

// Boolean creates a boolean schema.
func Boolean() *Schema {
	return &Schema{Type: "boolean"}
}

// Null creates a null schema.
func Null() *Schema {
	return &Schema{Type: "null"}
}

// Nullable makes a schema nullable by wrapping it in anyOf with null.
func Nullable(s *Schema) *Schema {
	return &Schema{
		AnyOf: []*Schema{s, Null()},
	}
}

// Describe adds a description to a schema.
func Describe(s *Schema, description string) *Schema {
	s.Description = description
	return s
}

// DefaultSchemaURI is the default JSON Schema URI used for generated schemas.
const DefaultSchemaURI = "https://json-schema.org/draft/2020-12/schema"

// FromType generates a JSON Schema from a Go type.
// Supports basic types, structs, slices, and maps.
// Adds $schema URI to root object schemas.
func FromType(v any) (*Schema, error) {
	s, err := fromReflectType(reflect.TypeOf(v))
	if err != nil {
		return nil, err
	}
	// Add $schema to root object schemas
	if s.Type == "object" {
		s.SchemaURI = DefaultSchemaURI
	}
	return s, nil
}

// MustFromType generates a JSON Schema from a Go type, panicking on error.
func MustFromType(v any) *Schema {
	s, err := FromType(v)
	if err != nil {
		panic(err)
	}
	return s
}

func fromReflectType(t reflect.Type) (*Schema, error) {
	if t == nil {
		return Null(), nil
	}

	// Handle pointers
	if t.Kind() == reflect.Ptr {
		s, err := fromReflectType(t.Elem())
		if err != nil {
			return nil, err
		}
		return Nullable(s), nil
	}

	switch t.Kind() {
	case reflect.String:
		return String(), nil

	case reflect.Bool:
		return Boolean(), nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		// Use "number" instead of "integer"
		return Number(), nil

	case reflect.Float32, reflect.Float64:
		return Number(), nil

	case reflect.Slice, reflect.Array:
		itemSchema, err := fromReflectType(t.Elem())
		if err != nil {
			return nil, err
		}
		return Array(itemSchema), nil

	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("map keys must be strings, got %s", t.Key().Kind())
		}
		valueSchema, err := fromReflectType(t.Elem())
		if err != nil {
			return nil, err
		}
		return &Schema{
			Type:                 "object",
			AdditionalProperties: nil, // Allow additional properties
			Items:                valueSchema,
		}, nil

	case reflect.Struct:
		return fromStructType(t)

	case reflect.Interface:
		return &Schema{}, nil // Any type

	default:
		return nil, fmt.Errorf("unsupported type: %s", t.Kind())
	}
}

func fromStructType(t reflect.Type) (*Schema, error) {
	properties := make(map[string]*Schema)
	required := make([]string, 0)

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get JSON tag
		name := field.Name
		jsonTag := field.Tag.Get("json")
		if jsonTag != "" {
			if jsonTag == "-" {
				continue
			}
			// Parse json tag (name,options)
			for i, c := range jsonTag {
				if c == ',' {
					jsonTag = jsonTag[:i]
					break
				}
			}
			if jsonTag != "" {
				name = jsonTag
			}
		}

		// Generate schema for field type
		fieldSchema, err := fromReflectType(field.Type)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", field.Name, err)
		}

		// Add description from tag
		if desc := field.Tag.Get("description"); desc != "" {
			fieldSchema.Description = desc
		}

		// Add enum from tag (comma-separated values)
		if enumTag := field.Tag.Get("enum"); enumTag != "" {
			enumValues := splitTag(enumTag)
			fieldSchema.Enum = make([]any, len(enumValues))
			for i, v := range enumValues {
				fieldSchema.Enum[i] = v
			}
		}

		// Add default from tag
		if defaultTag := field.Tag.Get("default"); defaultTag != "" {
			fieldSchema.Default = defaultTag
		}

		// Add itemRef for array items (sets ref on the items schema)
		if itemRefTag := field.Tag.Get("itemRef"); itemRefTag != "" {
			if fieldSchema.Items != nil {
				fieldSchema.Items.CustomRef = itemRefTag
			}
		}

		properties[name] = fieldSchema

		// Check if required (not a pointer and no omitempty)
		isOptional := field.Type.Kind() == reflect.Ptr ||
			(jsonTag != "" && containsOmitempty(field.Tag.Get("json")))
		if !isOptional {
			required = append(required, name)
		}
	}

	additionalProperties := false
	return &Schema{
		Type:                 "object",
		Properties:           properties,
		Required:             required,
		AdditionalProperties: &additionalProperties,
	}, nil
}

func containsOmitempty(tag string) bool {
	for _, part := range splitTag(tag) {
		if part == "omitempty" {
			return true
		}
	}
	return false
}

func splitTag(tag string) []string {
	var parts []string
	start := 0
	for i, c := range tag {
		if c == ',' {
			if start < i {
				parts = append(parts, tag[start:i])
			}
			start = i + 1
		}
	}
	if start < len(tag) {
		parts = append(parts, tag[start:])
	}
	return parts
}

// ToStrict modifies a schema to be compatible with strict mode:
// 1. Adds "additionalProperties": false to all object types
// 2. Ensures all properties are listed in "required"
func ToStrict(s *Schema) *Schema {
	if s == nil {
		return nil
	}

	// Deep copy and modify
	result := *s

	if result.Type == "object" {
		additionalProperties := false
		result.AdditionalProperties = &additionalProperties

		if result.Properties != nil {
			// Ensure all properties are required (sorted for deterministic output)
			allKeys := make([]string, 0, len(result.Properties))
			newProps := make(map[string]*Schema, len(result.Properties))
			for name, prop := range result.Properties {
				allKeys = append(allKeys, name)
				newProps[name] = ToStrict(prop)
			}
			sort.Strings(allKeys)
			result.Properties = newProps
			result.Required = allKeys
		}
	}

	if result.Type == "array" && result.Items != nil {
		result.Items = ToStrict(result.Items)
	}

	// Process composite schemas
	if result.AnyOf != nil {
		newAnyOf := make([]*Schema, len(result.AnyOf))
		for i, sub := range result.AnyOf {
			newAnyOf[i] = ToStrict(sub)
		}
		result.AnyOf = newAnyOf
	}

	if result.OneOf != nil {
		newOneOf := make([]*Schema, len(result.OneOf))
		for i, sub := range result.OneOf {
			newOneOf[i] = ToStrict(sub)
		}
		result.OneOf = newOneOf
	}

	if result.AllOf != nil {
		newAllOf := make([]*Schema, len(result.AllOf))
		for i, sub := range result.AllOf {
			newAllOf[i] = ToStrict(sub)
		}
		result.AllOf = newAllOf
	}

	return &result
}
