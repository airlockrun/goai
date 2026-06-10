package openai

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/tool"
)

// Chat Completions API request types

type chatRequest struct {
	Model           string              `json:"model"`
	Messages        []chatMessage       `json:"messages"`
	Stream          bool                `json:"stream"`
	Temperature     *float64            `json:"temperature,omitempty"`
	TopP            *float64            `json:"top_p,omitempty"`
	MaxTokens       *int                `json:"max_tokens,omitempty"`
	Stop            []string            `json:"stop,omitempty"`
	Tools           []chatTool          `json:"tools,omitempty"`
	ToolChoice      any                 `json:"tool_choice,omitempty"`
	ResponseFormat  *chatResponseFormat `json:"response_format,omitempty"`
	StreamOptions   *chatStreamOptions  `json:"stream_options,omitempty"`
	ReasoningEffort string              `json:"reasoning_effort,omitempty"`

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
	Role string `json:"role"`
	// Content is always emitted (no omitempty). OpenAI requires content
	// to be a string for assistant messages without tool_calls (even an
	// empty string), and to be null for tool-only messages. nil → "null",
	// "" → empty string. ai-sdk #14950.
	Content    any            `json:"content"`
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
	// Error rides on a top-level stream frame when OpenAI returns HTTP 200 and
	// then fails (e.g. insufficient_quota). Surfaced as a terminating error
	// when it arrives before any output. ai-sdk #15922.
	Error *OpenAIErrorInfo `json:"error,omitempty"`
}

type chatChunkChoice struct {
	Index        int                `json:"index"`
	Delta        chatChunkDelta     `json:"delta"`
	FinishReason string             `json:"finish_reason,omitempty"`
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
			text := getTextFromContent(msg.Content)
			cm := chatMessage{Role: "assistant"}
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
			// content: null is only allowed when tool_calls is non-empty;
			// otherwise OpenAI rejects with "expected a string, got null"
			// (ai-sdk #14950).
			if len(cm.ToolCalls) > 0 && text == "" {
				cm.Content = nil
			} else {
				cm.Content = text
			}
			result = append(result, cm)
		case message.RoleTool:
			// Chat Completions only supports string tool results.
			// Error if the message contains a FilePart.
			for _, part := range msg.Content.Parts {
				if _, ok := part.(message.FilePart); ok {
					return nil, errors.New("openai chat completions does not support file parts in tool results")
				}
			}

			for _, part := range msg.Content.Parts {
				if tr, ok := part.(message.ToolResultPart); ok {
					result = append(result, chatMessage{
						Role:       "tool",
						Content:    message.ToolOutputWire(tr.Output),
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
		case message.FilePart:
			switch d := p.Data.(type) {
			case message.FileDataBytes:
				if strings.HasPrefix(p.MimeType, "image/") {
					result = append(result, chatContentPart{
						Type: "image_url",
						ImageURL: &chatImageURL{
							URL: "data:" + p.MimeType + ";base64," + d.Data,
						},
					})
				} else if p.MimeType == "application/pdf" {
					filename := p.Filename
					if filename == "" {
						filename = "document.pdf"
					}
					result = append(result, chatContentPart{
						Type: "file",
						File: &chatFile{
							Filename: filename,
							FileData: "data:application/pdf;base64," + d.Data,
						},
					})
				}
				// OpenAI chat only supports image and PDF file parts; other
				// byte types are skipped.
			case message.FileDataURL:
				if strings.HasPrefix(p.MimeType, "image/") {
					result = append(result, chatContentPart{
						Type: "image_url",
						ImageURL: &chatImageURL{
							URL: d.URL,
						},
					})
				}
				// Chat Completions accepts remote URLs only for images.
			}
			// FileDataText / FileDataReference are not representable on the
			// chat content surface; skip.
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
		if t.IsProviderTool() {
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
