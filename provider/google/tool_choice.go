package google

import "encoding/json"

// convertToolChoice translates goai's loose ToolChoice — bare strings or
// structured objects — into Gemini's toolConfig.functionCallingConfig shape.
//
// Mirrors ai-sdk's prepareTools handling
// (packages/google/src/google-prepare-tools.ts):
//   - "auto"      → {mode: "AUTO"}
//   - "required"  → {mode: "ANY"}
//   - "none"      → {mode: "NONE"}
//   - {type: "tool", toolName: X} → {mode: "ANY", allowedFunctionNames: [X]}
//
// The bare-string and ai-sdk discriminated-union shapes are accepted in
// addition to the wire-form passthrough. Gemini does not have a "VALIDATED"
// strict-tools mode in goai's surface yet, so it is omitted.
func convertToolChoice(raw any) *geminiToolConfig {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case string:
		return toolConfigFromString(v)
	case map[string]any:
		return toolConfigFromMap(v)
	case map[string]string:
		m := make(map[string]any, len(v))
		for k, s := range v {
			m[k] = s
		}
		return toolConfigFromMap(m)
	default:
		b, err := json.Marshal(raw)
		if err != nil {
			return nil
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil && m != nil {
			return toolConfigFromMap(m)
		}
		return nil
	}
}

func toolConfigFromString(s string) *geminiToolConfig {
	switch s {
	case "":
		return nil
	case "auto":
		return modeOnly("AUTO")
	case "required":
		return modeOnly("ANY")
	case "none":
		return modeOnly("NONE")
	default:
		// Bare tool name → force that function tool.
		return &geminiToolConfig{
			FunctionCallingConfig: &geminiFunctionCallingConfig{
				Mode:                 "ANY",
				AllowedFunctionNames: []string{s},
			},
		}
	}
}

func toolConfigFromMap(m map[string]any) *geminiToolConfig {
	t, _ := m["type"].(string)
	switch t {
	case "auto":
		return modeOnly("AUTO")
	case "required":
		return modeOnly("ANY")
	case "none":
		return modeOnly("NONE")
	case "tool":
		name, _ := m["toolName"].(string)
		if name == "" {
			name, _ = m["name"].(string)
		}
		if name == "" {
			return modeOnly("ANY")
		}
		return &geminiToolConfig{
			FunctionCallingConfig: &geminiFunctionCallingConfig{
				Mode:                 "ANY",
				AllowedFunctionNames: []string{name},
			},
		}
	default:
		// Unknown shape — ignore rather than emit a malformed toolConfig.
		return nil
	}
}

func modeOnly(mode string) *geminiToolConfig {
	return &geminiToolConfig{
		FunctionCallingConfig: &geminiFunctionCallingConfig{Mode: mode},
	}
}
