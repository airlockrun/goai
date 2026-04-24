package google

import "encoding/json"

// convertJSONSchemaToOpenAPI translates a JSON Schema 7 document to the
// OpenAPI 3.0 subset that Gemini accepts as responseSchema.
// Port of ai-sdk packages/google/src/convert-json-schema-to-openapi-schema.ts.
// Returns nil when the root schema represents an empty object.
func convertJSONSchemaToOpenAPI(schemaBytes json.RawMessage) json.RawMessage {
	if len(schemaBytes) == 0 {
		return nil
	}
	var decoded any
	if err := json.Unmarshal(schemaBytes, &decoded); err != nil {
		return nil
	}
	converted := convertSchemaNode(decoded, true)
	if converted == nil {
		return nil
	}
	out, err := json.Marshal(converted)
	if err != nil {
		return nil
	}
	return out
}

func convertSchemaNode(node any, isRoot bool) any {
	if node == nil {
		return nil
	}
	if _, ok := node.(bool); ok {
		return map[string]any{"type": "boolean", "properties": map[string]any{}}
	}

	obj, ok := node.(map[string]any)
	if !ok {
		return node
	}

	if isEmptyObjectSchema(obj) {
		if isRoot {
			return nil
		}
		if desc, hasDesc := obj["description"].(string); hasDesc && desc != "" {
			return map[string]any{"type": "object", "description": desc}
		}
		return map[string]any{"type": "object"}
	}

	result := map[string]any{}

	if desc, ok := obj["description"].(string); ok && desc != "" {
		result["description"] = desc
	}
	if req, ok := obj["required"]; ok {
		result["required"] = req
	}
	if f, ok := obj["format"].(string); ok && f != "" {
		result["format"] = f
	}
	if constVal, ok := obj["const"]; ok {
		result["enum"] = []any{constVal}
	}

	if t, ok := obj["type"]; ok {
		switch v := t.(type) {
		case string:
			result["type"] = v
		case []any:
			hasNull := false
			var nonNull []any
			for _, typ := range v {
				if ts, ok := typ.(string); ok && ts == "null" {
					hasNull = true
				} else {
					nonNull = append(nonNull, typ)
				}
			}
			switch {
			case len(nonNull) == 0:
				result["type"] = "null"
			default:
				anyOf := make([]any, 0, len(nonNull))
				for _, typ := range nonNull {
					anyOf = append(anyOf, map[string]any{"type": typ})
				}
				result["anyOf"] = anyOf
				if hasNull {
					result["nullable"] = true
				}
			}
		}
	}

	if enumVals, ok := obj["enum"]; ok {
		result["enum"] = enumVals
	}

	if props, ok := obj["properties"].(map[string]any); ok {
		converted := make(map[string]any, len(props))
		for k, v := range props {
			converted[k] = convertSchemaNode(v, false)
		}
		result["properties"] = converted
	}

	if items, ok := obj["items"]; ok {
		switch it := items.(type) {
		case []any:
			arr := make([]any, len(it))
			for i, el := range it {
				arr[i] = convertSchemaNode(el, false)
			}
			result["items"] = arr
		default:
			result["items"] = convertSchemaNode(it, false)
		}
	}

	if allOf, ok := obj["allOf"].([]any); ok {
		arr := make([]any, len(allOf))
		for i, el := range allOf {
			arr[i] = convertSchemaNode(el, false)
		}
		result["allOf"] = arr
	}

	if anyOf, ok := obj["anyOf"].([]any); ok {
		nullIndex := -1
		for i, el := range anyOf {
			if m, ok := el.(map[string]any); ok {
				if t, ok := m["type"].(string); ok && t == "null" {
					nullIndex = i
					break
				}
			}
		}
		if nullIndex >= 0 {
			nonNull := make([]any, 0, len(anyOf)-1)
			for i, el := range anyOf {
				if i == nullIndex {
					continue
				}
				nonNull = append(nonNull, el)
			}
			if len(nonNull) == 1 {
				converted := convertSchemaNode(nonNull[0], false)
				if convMap, ok := converted.(map[string]any); ok {
					result["nullable"] = true
					for k, v := range convMap {
						result[k] = v
					}
				}
			} else {
				arr := make([]any, len(nonNull))
				for i, el := range nonNull {
					arr[i] = convertSchemaNode(el, false)
				}
				result["anyOf"] = arr
				result["nullable"] = true
			}
		} else {
			arr := make([]any, len(anyOf))
			for i, el := range anyOf {
				arr[i] = convertSchemaNode(el, false)
			}
			result["anyOf"] = arr
		}
	}

	if oneOf, ok := obj["oneOf"].([]any); ok {
		arr := make([]any, len(oneOf))
		for i, el := range oneOf {
			arr[i] = convertSchemaNode(el, false)
		}
		result["oneOf"] = arr
	}

	if minLen, ok := obj["minLength"]; ok {
		result["minLength"] = minLen
	}

	return result
}

func isEmptyObjectSchema(obj map[string]any) bool {
	t, ok := obj["type"].(string)
	if !ok || t != "object" {
		return false
	}
	props, hasProps := obj["properties"].(map[string]any)
	if hasProps && len(props) > 0 {
		return false
	}
	if ap, has := obj["additionalProperties"]; has && ap != nil && ap != false {
		return false
	}
	return true
}
