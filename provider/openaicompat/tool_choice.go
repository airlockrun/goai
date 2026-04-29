package openaicompat

import "encoding/json"

// convertToolChoice translates goai's loose ToolChoice — bare strings or
// structured objects — into the OpenAI Chat Completions tool_choice shape that
// most OpenAI-compatible providers (groq, mistral, deepseek, perplexity,
// fireworks, deepinfra, togetherai, cerebras, cohere) accept.
//
// Mirrors ai-sdk's prepareTools handling
// (packages/openai/src/chat/openai-chat-prepare-tools.ts):
//   - "auto" / "none" / "required" → bare string
//   - {type: "tool", toolName: X}  → {type: "function", function: {name: X}}
//
// Wire-form objects (e.g. {type: "function", function: {name: ...}}) pass
// through unchanged.
func convertToolChoice(raw any) any {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case string:
		switch v {
		case "":
			return nil
		case "auto", "none", "required":
			return v
		default:
			return functionToolChoice(v)
		}
	case map[string]any:
		return normalizeToolChoiceMap(v)
	case map[string]string:
		m := make(map[string]any, len(v))
		for k, s := range v {
			m[k] = s
		}
		return normalizeToolChoiceMap(m)
	default:
		b, err := json.Marshal(raw)
		if err != nil {
			return raw
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil && m != nil {
			return normalizeToolChoiceMap(m)
		}
		return raw
	}
}

func normalizeToolChoiceMap(m map[string]any) any {
	t, _ := m["type"].(string)
	switch t {
	case "auto", "none", "required":
		return t
	case "tool":
		name, _ := m["toolName"].(string)
		if name == "" {
			name, _ = m["name"].(string)
		}
		if name == "" {
			return m
		}
		return functionToolChoice(name)
	case "function":
		return m
	default:
		return m
	}
}

func functionToolChoice(name string) any {
	return map[string]any{
		"type":     "function",
		"function": map[string]any{"name": name},
	}
}
