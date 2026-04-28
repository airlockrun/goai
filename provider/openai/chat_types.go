package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/tool"
)

// Chat Completions API request types

type chatRequest struct {
	Model          string              `json:"model"`
	Messages       []chatMessage       `json:"messages"`
	Stream         bool                `json:"stream"`
	Temperature    *float64            `json:"temperature,omitempty"`
	TopP           *float64            `json:"top_p,omitempty"`
	MaxTokens      *int                `json:"max_tokens,omitempty"`
	Stop           []string            `json:"stop,omitempty"`
	Tools          []chatTool          `json:"tools,omitempty"`
	ToolChoice     any                 `json:"tool_choice,omitempty"`
	ResponseFormat *chatResponseFormat `json:"response_format,omitempty"`
	StreamOptions  *chatStreamOptions  `json:"stream_options,omitempty"`

	Logprobs    *bool `json:"logprobs,omitempty"`
	TopLogprobs *int  `json:"top_logprobs,omitempty"`
}

// chatResponseFormat configures structured output for chat completions.
// Mirrors ai-sdk openai-chat-language-model.ts:147-160.
type chatResponseFormat struct {
	Type       string          `json:"type"` // "json_object" or "json_schema"
	JSONSchema *chatJSONSchema `json:"json_schema,omitempty"`
}

type chatJSONSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema"`
	Strict      *bool           `json:"strict,omitempty"`
}

type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
	File     *chatFile     `json:"file,omitempty"`
}

type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type chatFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
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

// Chat Completions API response types

type chatCompletionChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChunkChoice `json:"choices"`
	Usage   *chatUsage        `json:"usage,omitempty"`
}

type chatChunkChoice struct {
	Index        int              `json:"index"`
	Delta        chatChunkDelta   `json:"delta"`
	FinishReason string           `json:"finish_reason,omitempty"`
	Logprobs     *chatChunkLogprobs `json:"logprobs,omitempty"`
}

// chatChunkLogprobs mirrors OpenAI's streaming logprobs shape.
// Surfaced via providerMetadata.openai.logprobs.
type chatChunkLogprobs struct {
	Content []chatLogprobToken `json:"content,omitempty"`
}

type chatLogprobToken struct {
	Token       string              `json:"token"`
	Logprob     float64             `json:"logprob"`
	TopLogprobs []chatLogprobAltTok `json:"top_logprobs,omitempty"`
}

type chatLogprobAltTok struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
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
}

// Conversion functions

func convertToChatMessages(messages []message.Message) ([]chatMessage, error) {
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
			cm := chatMessage{
				Role:    "assistant",
				Content: getTextFromContent(msg.Content),
			}
			// Add tool calls if present. Empty arguments serialize as "{}"
			// so Chat Completions never sees a blank string (ai-sdk #953385d).
			for _, part := range msg.Content.Parts {
				if tc, ok := part.(message.ToolCallPart); ok {
					args := string(tc.Input)
					if args == "" {
						args = "{}"
					}
					cm.ToolCalls = append(cm.ToolCalls, chatToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: chatFunctionCall{
							Name:      tc.Name,
							Arguments: args,
						},
					})
				}
			}
			result = append(result, cm)
		case message.RoleTool:
			// Chat Completions only supports string tool results.
			// Error if the message contains ImagePart or FilePart.
			for _, part := range msg.Content.Parts {
				switch part.(type) {
				case message.ImagePart:
					return nil, fmt.Errorf("openai chat completions does not support image parts in tool results")
				case message.FilePart:
					return nil, fmt.Errorf("openai chat completions does not support file parts in tool results")
				}
			}

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

	return result, nil
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
			imageURL := p.Image
			if !strings.HasPrefix(imageURL, "http") && !strings.HasPrefix(imageURL, "data:") {
				mime := p.MimeType
				if mime == "" {
					mime = "image/png"
				}
				imageURL = "data:" + mime + ";base64," + imageURL
			}
			result = append(result, chatContentPart{
				Type: "image_url",
				ImageURL: &chatImageURL{
					URL: imageURL,
				},
			})
		case message.FilePart:
			if p.MimeType == "application/pdf" {
				filename := p.Filename
				if filename == "" {
					filename = "document.pdf"
				}
				result = append(result, chatContentPart{
					Type: "file",
					File: &chatFile{
						Filename: filename,
						FileData: "data:application/pdf;base64," + p.Data,
					},
				})
			}
			// OpenAI only supports PDF file parts; other types are silently skipped.
		}
	}
	return result
}

func convertToChatTools(tools []tool.Tool) []chatTool {
	result := make([]chatTool, 0, len(tools))

	for _, t := range tools {
		// Chat Completions only supports function-type tools. Provider-defined
		// hosted tools (web_search, custom, tool_search, ...) are only supported
		// on the Responses API; ai-sdk's prepareChatTools emits an 'unsupported'
		// warning and drops them.
		if t.Type == "provider" {
			continue
		}
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
