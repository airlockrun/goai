package cohere

import (
	"encoding/json"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/tool"
)

// Request types

type cohereRequest struct {
	Model          string                `json:"model"`
	Message        string                `json:"message"`
	ChatHistory    []cohereMessage       `json:"chat_history,omitempty"`
	Preamble       string                `json:"preamble,omitempty"`
	Temperature    *float64              `json:"temperature,omitempty"`
	P              *float64              `json:"p,omitempty"`
	K              *int                  `json:"k,omitempty"`
	MaxTokens      *int                  `json:"max_tokens,omitempty"`
	Stream         bool                  `json:"stream"`
	Tools          []cohereTool          `json:"tools,omitempty"`
	Thinking       *cohereThinking       `json:"thinking,omitempty"`
	ResponseFormat *cohereResponseFormat `json:"response_format,omitempty"`
}

// cohereResponseFormat mirrors Cohere's response_format. Unlike OpenAI,
// Cohere uses type "json_object" even when a schema is supplied.
type cohereResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// cohereThinking configures reasoning features.
type cohereThinking struct {
	Type        string `json:"type,omitempty"`         // "enabled" or "disabled"
	TokenBudget int    `json:"token_budget,omitempty"` // Max tokens for thinking
}

type cohereMessage struct {
	Role    string `json:"role"`
	Message string `json:"message"`
}

type cohereTool struct {
	Name                 string                     `json:"name"`
	Description          string                     `json:"description"`
	ParameterDefinitions map[string]cohereToolParam `json:"parameter_definitions,omitempty"`
}

type cohereToolParam struct {
	Description string `json:"description,omitempty"`
	Type        string `json:"type"`
	Required    bool   `json:"required,omitempty"`
}

// Stream event types

type cohereStreamEvent struct {
	EventType    string           `json:"event_type"`
	Text         string           `json:"text,omitempty"`
	ToolCalls    []cohereToolCall `json:"tool_calls,omitempty"`
	FinishReason string           `json:"finish_reason,omitempty"`
	Response     *cohereResponse  `json:"response,omitempty"`
}

type cohereToolCall struct {
	Name       string         `json:"name"`
	Parameters map[string]any `json:"parameters"`
}

type cohereResponse struct {
	Meta *cohereMeta `json:"meta,omitempty"`
}

type cohereMeta struct {
	Tokens *cohereTokens `json:"tokens,omitempty"`
}

type cohereTokens struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Conversion functions

func getTextFromContent(content message.Content) string {
	if content.Text != "" {
		return content.Text
	}
	for _, part := range content.Parts {
		if tp, ok := part.(message.TextPart); ok {
			return tp.Text
		}
	}
	return ""
}

func convertToCohereTools(tools []tool.Tool) []cohereTool {
	result := make([]cohereTool, 0, len(tools))

	for _, t := range tools {
		ct := cohereTool{
			Name:        t.Name,
			Description: t.Description,
		}

		// Parse JSON schema to extract parameter definitions
		if len(t.InputSchema) > 0 {
			var schema struct {
				Properties map[string]struct {
					Type        string `json:"type"`
					Description string `json:"description"`
				} `json:"properties"`
				Required []string `json:"required"`
			}
			if err := json.Unmarshal(t.InputSchema, &schema); err == nil {
				requiredSet := make(map[string]bool)
				for _, r := range schema.Required {
					requiredSet[r] = true
				}

				ct.ParameterDefinitions = make(map[string]cohereToolParam)
				for pname, prop := range schema.Properties {
					ct.ParameterDefinitions[pname] = cohereToolParam{
						Type:        prop.Type,
						Description: prop.Description,
						Required:    requiredSet[pname],
					}
				}
			}
		}

		result = append(result, ct)
	}

	return result
}
