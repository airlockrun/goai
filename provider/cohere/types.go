package cohere

import (
	"encoding/json"
	"github.com/airlockrun/goai/tool"
)

type cohereRequest struct {
	Model            string                `json:"model"`
	Messages         []any                 `json:"messages"`
	Documents        []any                 `json:"documents,omitempty"`
	Stream           bool                  `json:"stream"`
	Temperature      *float64              `json:"temperature,omitempty"`
	P                *float64              `json:"p,omitempty"`
	K                *int                  `json:"k,omitempty"`
	MaxTokens        *int                  `json:"max_tokens,omitempty"`
	Seed             *int                  `json:"seed,omitempty"`
	StopSequences    []string              `json:"stop_sequences,omitempty"`
	FrequencyPenalty *float64              `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64              `json:"presence_penalty,omitempty"`
	Tools            []cohereTool          `json:"tools,omitempty"`
	ToolChoice       string                `json:"tool_choice,omitempty"`
	Thinking         *cohereThinking       `json:"thinking,omitempty"`
	ResponseFormat   *cohereResponseFormat `json:"response_format,omitempty"`
}

type cohereResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}
type cohereThinking struct {
	Type        string `json:"type,omitempty"`
	TokenBudget int    `json:"token_budget,omitempty"`
}
type cohereTool struct {
	Type     string         `json:"type"`
	Function cohereFunction `json:"function"`
}
type cohereFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

func convertToCohereTools(tools []tool.Tool) []cohereTool {
	out := make([]cohereTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, cohereTool{Type: "function", Function: cohereFunction{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}})
	}
	return out
}

type cohereStreamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		FinishReason string          `json:"finish_reason"`
		Usage        json.RawMessage `json:"usage"`
		Message      struct {
			Content struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			} `json:"content"`
			ToolPlan  string `json:"tool_plan"`
			ToolCalls struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"delta"`
}
