package openai

import (
	"encoding/json"

	"github.com/airlockrun/goai/tool"
)

// convertChatToolChoice translates goai's loose ToolChoice — bare strings or
// structured objects — into OpenAI Chat Completions' tool_choice shape.
//
// Mirrors ai-sdk's prepareTools handling
// (packages/openai/src/chat/openai-chat-prepare-tools.ts):
//   - "auto" / "none" / "required" → bare string (API accepts as-is)
//   - {type: "tool", toolName: X}  → {type: "function", function: {name: X}}
//
// Already-wire-form objects (e.g. {type: "function", function: {name: ...}})
// pass through unchanged.
func convertChatToolChoice(raw any) any {
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
			// Bare tool name → force that function tool.
			return chatFunctionToolChoice(v)
		}
	case map[string]any:
		return normalizeChatToolChoiceMap(v)
	case map[string]string:
		m := make(map[string]any, len(v))
		for k, s := range v {
			m[k] = s
		}
		return normalizeChatToolChoiceMap(m)
	default:
		b, err := json.Marshal(raw)
		if err != nil {
			return raw
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil && m != nil {
			return normalizeChatToolChoiceMap(m)
		}
		return raw
	}
}

func normalizeChatToolChoiceMap(m map[string]any) any {
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
		return chatFunctionToolChoice(name)
	case "function":
		// Already wire-form. Pass through.
		return m
	default:
		return m
	}
}

func chatFunctionToolChoice(name string) any {
	return map[string]any{
		"type":     "function",
		"function": map[string]any{"name": name},
	}
}

// convertResponsesToolChoice translates goai's loose ToolChoice — bare strings
// ("auto" / "none" / "required" / a tool name) or structured objects — into the
// OpenAI Responses API's accepted shapes.
//
// Mirrors ai-sdk's prepareResponsesTools tool_choice handling
// (packages/openai/src/responses/openai-responses-prepare-tools.ts):
//   - "auto" / "none" / "required" → bare string (API accepts as-is)
//   - {type: "tool", toolName: X}  → {type: "function", name: X} for normal
//     tools, {type: <wire-type>} for hosted tools (web_search / tool_search),
//     {type: "custom", name: X} for openai.custom tools.
//
// Already-wire-form objects (e.g. {type: "function", name: ...}, {type: "web_search"})
// are passed through unchanged. Tool lookup uses options.Tools so the function
// can pick the right hosted-vs-function shape from the tool name.
func convertResponsesToolChoice(raw any, tools []tool.Tool) any {
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
			// Bare tool name → resolve to function/hosted/custom.
			return resolveResponsesToolChoiceByName(v, tools)
		}
	case map[string]any:
		return normalizeResponsesToolChoiceMap(v, tools)
	case map[string]string:
		m := make(map[string]any, len(v))
		for k, s := range v {
			m[k] = s
		}
		return normalizeResponsesToolChoiceMap(m, tools)
	default:
		b, err := json.Marshal(raw)
		if err != nil {
			return raw
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil && m != nil {
			return normalizeResponsesToolChoiceMap(m, tools)
		}
		return raw
	}
}

func normalizeResponsesToolChoiceMap(m map[string]any, tools []tool.Tool) any {
	t, _ := m["type"].(string)
	switch t {
	case "auto", "none", "required":
		// ai-sdk shape {type: "auto"} → bare string for OpenAI Responses.
		return t
	case "tool":
		name, _ := m["toolName"].(string)
		if name == "" {
			name, _ = m["name"].(string)
		}
		if name == "" {
			return m
		}
		return resolveResponsesToolChoiceByName(name, tools)
	default:
		// Already wire-form (e.g. {type: "function", name: X},
		// {type: "web_search"}); pass through.
		return m
	}
}

func resolveResponsesToolChoiceByName(name string, tools []tool.Tool) any {
	// Hard-coded hosted wire-type literals — matches ai-sdk's check against
	// 'web_search'/'code_interpreter'/etc. Hosted tools in goai often have
	// an empty Name field (only ProviderID is set) so a bare-name path keeps
	// the API ergonomic when the caller knows the wire type.
	switch name {
	case "web_search", "tool_search":
		return map[string]any{"type": name}
	}
	// Look up via tools[] — pick the right shape based on the tool kind.
	for _, t := range tools {
		if t.Name != name && t.ProviderID != name {
			continue
		}
		if t.Type == "provider" {
			switch t.ProviderID {
			case ToolIDWebSearch:
				return map[string]any{"type": "web_search"}
			case ToolIDToolSearch:
				return map[string]any{"type": "tool_search"}
			case ToolIDCustom:
				return map[string]any{"type": "custom", "name": t.Name}
			}
		}
		break
	}
	// Default: a regular function tool.
	return map[string]any{"type": "function", "name": name}
}
