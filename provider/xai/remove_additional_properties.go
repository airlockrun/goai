package xai

import "encoding/json"

// removeAdditionalPropertiesFalse strips `additionalProperties: false`
// entries from a JSON schema, recursing through nested objects and arrays
// while leaving non-boolean additionalProperties schemas intact. xAI
// rejects `additionalProperties: false` on tool schemas that declare an
// explicit `properties` key; its docs treat it as false by default, so
// dropping it is safe.
// Mirrors ai-sdk packages/xai/src/remove-additional-properties.ts.
func removeAdditionalPropertiesFalse(value any) any {
	switch v := value.(type) {
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = removeAdditionalPropertiesFalse(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, propertyValue := range v {
			if key == "additionalProperties" {
				if b, ok := propertyValue.(bool); ok && !b {
					continue
				}
			}
			out[key] = removeAdditionalPropertiesFalse(propertyValue)
		}
		return out
	default:
		return value
	}
}

// sanitizeToolSchema applies removeAdditionalPropertiesFalse to a raw
// JSON-schema payload. A nil or unparseable schema is returned unchanged.
func sanitizeToolSchema(schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 {
		return schema
	}
	var decoded any
	if err := json.Unmarshal(schema, &decoded); err != nil {
		return schema
	}
	cleaned, err := json.Marshal(removeAdditionalPropertiesFalse(decoded))
	if err != nil {
		return schema
	}
	return cleaned
}
