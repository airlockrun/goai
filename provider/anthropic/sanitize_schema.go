package anthropic

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Anthropic's structured-output decoder only accepts a strict subset of
// JSON Schema; passing common validation keywords like `minimum`,
// `maxLength`, etc. yields a 400. sanitizeJSONSchema strips those
// keywords from the schema sent to Anthropic and folds them back into
// the schema's `description` so the model still sees the constraint.
// AI SDK result validation (caller-side) keeps the original schema.
//
// Mirrors ai-sdk #14790 (packages/anthropic/src/sanitize-json-schema.ts).

var supportedStringFormats = map[string]struct{}{
	"date-time": {},
	"time":      {},
	"date":      {},
	"duration":  {},
	"email":     {},
	"hostname":  {},
	"uri":       {},
	"ipv4":      {},
	"ipv6":      {},
	"uuid":      {},
}

// constraintKeys are JSON-Schema keywords Anthropic rejects. We extract
// their values into the description and drop them from the wire schema.
// Order is preserved so the description text is stable.
var constraintKeys = []string{
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"minLength",
	"maxLength",
	"pattern",
	"minItems",
	"maxItems",
	"uniqueItems",
	"minProperties",
	"maxProperties",
	"not",
}

// sanitizeJSONSchema returns a sanitized copy of schema. When schema is
// not a JSON object (e.g. nil or a `true`/`false` schema), it's returned
// unchanged.
func sanitizeJSONSchema(schema json.RawMessage) (json.RawMessage, error) {
	if len(schema) == 0 {
		return schema, nil
	}
	var any any
	if err := json.Unmarshal(schema, &any); err != nil {
		return nil, fmt.Errorf("sanitize schema: %w", err)
	}
	clean := sanitizeNode(any)
	out, err := json.Marshal(clean)
	if err != nil {
		return nil, fmt.Errorf("sanitize schema: %w", err)
	}
	return out, nil
}

// sanitizeNode walks any JSON value, sanitizing object schemas. Booleans
// (the JSON Schema `true`/`false` shorthand) and primitives pass through.
func sanitizeNode(node any) any {
	switch v := node.(type) {
	case map[string]any:
		return sanitizeObject(v)
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = sanitizeNode(item)
		}
		return out
	default:
		return v
	}
}

func sanitizeObject(in map[string]any) map[string]any {
	// $ref objects are kept as-is (siblings would be ignored anyway).
	if ref, ok := in["$ref"].(string); ok && ref != "" {
		return map[string]any{"$ref": ref}
	}

	out := map[string]any{}

	// Identity / annotation keys pass through.
	for _, k := range []string{"$schema", "$id", "title", "description", "type"} {
		if v, ok := in[k]; ok {
			out[k] = v
		}
	}
	// Optional keys preserved when present.
	if v, ok := in["default"]; ok {
		out["default"] = v
	}
	if v, ok := in["const"]; ok {
		out["const"] = v
	}
	if v, ok := in["enum"]; ok {
		out["enum"] = v
	}

	// anyOf / oneOf collapse to anyOf — Anthropic's decoder doesn't
	// distinguish, mirror ai-sdk.
	if items, ok := in["anyOf"].([]any); ok {
		out["anyOf"] = sanitizeDefList(items)
	} else if items, ok := in["oneOf"].([]any); ok {
		out["anyOf"] = sanitizeDefList(items)
	}
	if items, ok := in["allOf"].([]any); ok {
		out["allOf"] = sanitizeDefList(items)
	}
	if defs, ok := in["definitions"].(map[string]any); ok {
		out["definitions"] = sanitizeDefMap(defs)
	}
	if defs, ok := in["$defs"].(map[string]any); ok {
		out["$defs"] = sanitizeDefMap(defs)
	}

	// Object schemas: properties, required, additionalProperties=false.
	hasProperties := false
	if props, ok := in["properties"].(map[string]any); ok {
		out["properties"] = sanitizeDefMap(props)
		hasProperties = true
	}
	if t, _ := in["type"].(string); t == "object" || hasProperties {
		out["additionalProperties"] = false
		if req, ok := in["required"]; ok {
			out["required"] = req
		}
	}

	// Array items: schema or tuple of schemas.
	if items, ok := in["items"]; ok {
		switch its := items.(type) {
		case []any:
			out["items"] = sanitizeDefList(its)
		default:
			out["items"] = sanitizeNode(its)
		}
	}

	// Format: only a small whitelist passes through; everything else
	// becomes part of the constraint description.
	if f, ok := in["format"].(string); ok {
		if _, supported := supportedStringFormats[f]; supported {
			out["format"] = f
		}
	}

	if extra := constraintDescription(in); extra != "" {
		if d, ok := out["description"].(string); ok && d != "" {
			out["description"] = d + "\n" + extra
		} else {
			out["description"] = extra
		}
	}

	return out
}

func sanitizeDefList(in []any) []any {
	out := make([]any, len(in))
	for i, item := range in {
		out[i] = sanitizeNode(item)
	}
	return out
}

func sanitizeDefMap(in map[string]any) map[string]any {
	keys := make([]string, 0, len(in))
	for k := range in {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(map[string]any, len(in))
	for _, k := range keys {
		out[k] = sanitizeNode(in[k])
	}
	return out
}

// constraintDescription returns "minimum: 0; maxLength: 100." for the
// keywords we stripped, plus any non-whitelisted format. Empty string
// when nothing was stripped.
func constraintDescription(schema map[string]any) string {
	parts := make([]string, 0, len(constraintKeys))
	for _, k := range constraintKeys {
		v, ok := schema[k]
		if !ok || v == nil {
			continue
		}
		if b, isBool := v.(bool); isBool && !b {
			continue
		}
		parts = append(parts, formatConstraintName(k)+": "+formatConstraintValue(v))
	}
	if f, ok := schema["format"].(string); ok && f != "" {
		if _, supported := supportedStringFormats[f]; !supported {
			parts = append(parts, "format: "+f)
		}
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "; ") + "."
}

func formatConstraintName(key string) string {
	var b strings.Builder
	for _, r := range key {
		if r >= 'A' && r <= 'Z' {
			b.WriteByte(' ')
			b.WriteRune(r + ('a' - 'A'))
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func formatConstraintValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}
