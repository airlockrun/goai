package anthropic

import "encoding/json"

// convertAnthropicToolChoice translates goai's loose ToolChoice — which can
// be a bare string ("auto" / "none" / "required" / a specific tool name) or a
// structured object — into Anthropic's wire-format `tool_choice`.
//
// Mirrors ai-sdk's prepareTools tool_choice handling
// (packages/anthropic/src/anthropic-prepare-tools.ts):
//   - "auto"     → {type: "auto"}
//   - "required" → {type: "any"}        (Anthropic uses "any", not "required")
//   - "none"     → nil + dropTools=true (Anthropic models "none" by sending no tools)
//   - "tool"     → {type: "tool", name: <name>}
//
// Already-wire-form maps (e.g. {type: "any"}, {type: "tool", name: "x"}) are
// passed through unchanged. ai-sdk-shaped maps with `toolName` are normalized
// to `name`.
//
// dropTools=true tells the caller to clear req.Tools (Anthropic has no
// "none" tool_choice; sending no tools is the only way to forbid tool use).
func convertAnthropicToolChoice(raw any) (choice any, dropTools bool) {
	if raw == nil {
		return nil, false
	}
	switch v := raw.(type) {
	case string:
		return convertAnthropicToolChoiceString(v)
	case map[string]any:
		return normalizeAnthropicToolChoiceMap(v)
	case map[string]string:
		m := make(map[string]any, len(v))
		for k, s := range v {
			m[k] = s
		}
		return normalizeAnthropicToolChoiceMap(m)
	default:
		// Round-trip through JSON for typed structs (e.g. callers using their
		// own ToolChoice struct). If that fails, hand the value back as-is —
		// the API will surface the error if it isn't a valid shape.
		b, err := json.Marshal(raw)
		if err != nil {
			return raw, false
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil && m != nil {
			return normalizeAnthropicToolChoiceMap(m)
		}
		return raw, false
	}
}

func convertAnthropicToolChoiceString(s string) (any, bool) {
	switch s {
	case "":
		return nil, false
	case "auto":
		return map[string]any{"type": "auto"}, false
	case "required":
		return map[string]any{"type": "any"}, false
	case "none":
		return nil, true
	default:
		// Bare tool name → force that tool.
		return map[string]any{"type": "tool", "name": s}, false
	}
}

func normalizeAnthropicToolChoiceMap(m map[string]any) (any, bool) {
	t, _ := m["type"].(string)
	switch t {
	case "auto":
		return map[string]any{"type": "auto"}, false
	case "required":
		return map[string]any{"type": "any"}, false
	case "any":
		// already wire form
		return m, false
	case "none":
		return nil, true
	case "tool":
		// ai-sdk shape uses `toolName`; Anthropic wire uses `name`. Accept both.
		name, _ := m["name"].(string)
		if name == "" {
			if tn, ok := m["toolName"].(string); ok {
				name = tn
			}
		}
		return map[string]any{"type": "tool", "name": name}, false
	default:
		// Unknown shape — pass through and let the API decide.
		return m, false
	}
}
