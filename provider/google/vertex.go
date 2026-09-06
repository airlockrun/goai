package google

import (
	"encoding/json"
	"github.com/airlockrun/goai/tool"
)

// ResponseSchema converts JSON Schema to Gemini's supported OpenAPI schema.
func ResponseSchema(schema json.RawMessage) json.RawMessage {
	return convertJSONSchemaToOpenAPI(schema)
}

// ThoughtSignature reads the Google-family metadata attached to a message part.
func ThoughtSignature(options map[string]any) string {
	for _, key := range []string{"googleVertex", "vertex", "google"} {
		if value, ok := options[key].(map[string]any); ok {
			if signature, ok := value["thoughtSignature"].(string); ok {
				return signature
			}
		}
	}
	signature, _ := options["thoughtSignature"].(string)
	return signature
}

// PrepareTools returns Gemini wire tools for Google and Vertex endpoints.
func PrepareTools(tools []tool.Tool, modelID string) any { return prepareGeminiTools(tools, modelID) }

// ToolConfig returns the common Gemini function-calling configuration.
func ToolConfig(choice any) map[string]any {
	config := convertToolChoice(choice)
	if config == nil {
		return nil
	}
	data, _ := json.Marshal(config)
	var result map[string]any
	_ = json.Unmarshal(data, &result)
	return result
}
