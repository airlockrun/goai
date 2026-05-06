package openaicompat

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Request types

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []any           `json:"messages"`
	Stream         bool            `json:"stream"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Tools          []chatTool      `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	StreamOptions  *streamOptions  `json:"stream_options,omitempty"`
}

// responseFormat mirrors OpenAI chat response_format.
// Type is "json_object" or "json_schema".
type responseFormat struct {
	Type       string              `json:"type"`
	JSONSchema *responseJSONSchema `json:"json_schema,omitempty"`
}

type responseJSONSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema"`
	Strict      *bool           `json:"strict,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role string `json:"role"`
	// Content is always emitted; OpenAI-compatible servers accept null
	// only when tool_calls is non-empty. ai-sdk #14950.
	Content    any            `json:"content"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatTool struct {
	Type     string          `json:"type"`
	Function chatFunctionDef `json:"function"`
}

type chatFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Response types

type chatCompletionChunk struct {
	ID       string            `json:"id"`
	Object   string            `json:"object"`
	Created  int64             `json:"created"`
	Model    string            `json:"model"`
	Choices  []chatChunkChoice `json:"choices"`
	UsageRaw json.RawMessage   `json:"usage,omitempty"`
}

type chatChunkChoice struct {
	Index        int            `json:"index"`
	Delta        chatChunkDelta `json:"delta"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

type chatChunkDelta struct {
	Role      string              `json:"role,omitempty"`
	Content   string              `json:"content,omitempty"`
	ToolCalls []chatChunkToolCall `json:"tool_calls,omitempty"`
}

type chatChunkToolCall struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function chatFunctionCall `json:"function"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// Cached-token sub-fields surfaced by various openai-compatible
	// providers. Mistral can send any of three shapes (ai-sdk #14889);
	// OpenAI itself uses prompt_tokens_details.cached_tokens.
	NumCachedTokens     int                  `json:"num_cached_tokens,omitempty"`
	PromptTokensDetails *promptTokensDetails `json:"prompt_tokens_details,omitempty"`
	PromptTokenDetails  *promptTokensDetails `json:"prompt_token_details,omitempty"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// usageFromChat builds a stream.Usage from chatUsage, expanding cached-
// token info into the Usage.InputTokens.{NoCache,CacheRead} breakdown
// when the server reports it. Mirrors ai-sdk #14889 (Mistral) and the
// equivalent OpenAI prompt_tokens_details handling.
func usageFromChat(u chatUsage) stream.Usage {
	cached := u.NumCachedTokens
	if cached == 0 && u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	if cached == 0 && u.PromptTokenDetails != nil {
		cached = u.PromptTokenDetails.CachedTokens
	}
	out := stream.Usage{
		InputTokens:  stream.InputTokens{Total: stream.IntPtr(u.PromptTokens)},
		OutputTokens: stream.OutputTokens{Total: stream.IntPtr(u.CompletionTokens), Text: stream.IntPtr(u.CompletionTokens)},
	}
	if cached > 0 {
		out.InputTokens.CacheRead = stream.IntPtr(cached)
		out.InputTokens.NoCache = stream.IntPtr(u.PromptTokens - cached)
	} else {
		out.InputTokens.NoCache = stream.IntPtr(u.PromptTokens)
	}
	return out
}

// Conversion functions

func convertToMessages(messages []message.Message) []chatMessage {
	result := make([]chatMessage, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			result = append(result, chatMessage{
				Role:    "system",
				Content: getTextFromContent(msg.Content),
			})
		case message.RoleUser:
			result = append(result, chatMessage{
				Role:    "user",
				Content: convertUserContent(msg.Content),
			})
		case message.RoleAssistant:
			text := getTextFromContent(msg.Content)
			cm := chatMessage{Role: "assistant"}
			for _, part := range msg.Content.Parts {
				if tc, ok := part.(message.ToolCallPart); ok {
					cm.ToolCalls = append(cm.ToolCalls, chatToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: chatFunctionCall{
							Name:      tc.Name,
							Arguments: string(tc.Input),
						},
					})
				}
			}
			if len(cm.ToolCalls) > 0 && text == "" {
				cm.Content = nil
			} else {
				cm.Content = text
			}
			result = append(result, cm)
		case message.RoleTool:
			for _, part := range msg.Content.Parts {
				if tr, ok := part.(message.ToolResultPart); ok {
					resultStr := ""
					switch v := tr.Result.(type) {
					case string:
						resultStr = v
					default:
						if b, err := json.Marshal(v); err == nil {
							resultStr = string(b)
						}
					}
					result = append(result, chatMessage{
						Role:       "tool",
						Content:    resultStr,
						ToolCallID: tr.ToolCallID,
					})
				}
			}
		}
	}

	return result
}

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

func convertUserContent(content message.Content) any {
	// If simple text, return it directly
	if content.Text != "" && len(content.Parts) == 0 {
		return content.Text
	}

	// Check if we have only text parts
	hasOnlyText := true
	for _, part := range content.Parts {
		if _, ok := part.(message.TextPart); !ok {
			hasOnlyText = false
			break
		}
	}

	if hasOnlyText && len(content.Parts) == 1 {
		if tp, ok := content.Parts[0].(message.TextPart); ok {
			return tp.Text
		}
	}

	// Build multipart content
	result := make([]chatContentPart, 0, len(content.Parts))
	for _, part := range content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			result = append(result, chatContentPart{
				Type: "text",
				Text: p.Text,
			})
		case message.ImagePart:
			result = append(result, chatContentPart{
				Type: "image_url",
				ImageURL: &chatImageURL{
					URL: p.Image,
				},
			})
		case message.FilePart:
			if strings.HasPrefix(p.MimeType, "text/") {
				decoded, err := base64.StdEncoding.DecodeString(p.Data)
				if err != nil {
					decoded = []byte(p.Data)
				}
				result = append(result, chatContentPart{
					Type: "text",
					Text: string(decoded),
				})
			}
		}
	}
	return result
}

func convertToTools(tools []tool.Tool) []chatTool {
	result := make([]chatTool, 0, len(tools))

	for _, t := range tools {
		result = append(result, chatTool{
			Type: "function",
			Function: chatFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}

	return result
}
