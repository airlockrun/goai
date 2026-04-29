package deepseek

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/message"
)

// convertMessages applies DeepSeek's reasoning_content rules on top of the
// standard openai-compatible chat-message conversion.
//
// Mirrors ai-sdk's convertToDeepSeekChatMessages (PR #14739):
//   - deepseek-v4 / deepseek-v4-pro: every assistant turn must echo back
//     reasoning_content (empty string when the source has no reasoning part).
//   - deepseek-reasoner (R1): prior-turn reasoning must be stripped — only
//     reasoning that follows the last user message is sent.
//   - deepseek-chat / deepseek-coder: no reasoning_content field at all.
func convertMessages(modelID string, messages []message.Message) ([]any, error) {
	isV4 := strings.Contains(modelID, "deepseek-v4")

	lastUserMsgIndex := -1
	for i, m := range messages {
		if m.Role == message.RoleUser {
			lastUserMsgIndex = i
		}
	}

	var out []any
	for i, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			out = append(out, map[string]any{
				"role":    "system",
				"content": getText(msg.Content),
			})

		case message.RoleUser:
			out = append(out, map[string]any{
				"role":    "user",
				"content": convertUserContent(msg.Content),
			})

		case message.RoleAssistant:
			text := ""
			var reasoning strings.Builder
			hasReasoning := false
			var toolCalls []map[string]any

			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.TextPart:
					text += p.Text
				case message.ReasoningPart:
					// R1 rejects prior reasoning; V4 requires it. Other
					// deepseek models drop the field entirely (handled
					// below by the !hasReasoning && !isV4 branch).
					if i <= lastUserMsgIndex && !isV4 {
						continue
					}
					reasoning.WriteString(p.Text)
					hasReasoning = true
				case message.ToolCallPart:
					toolCalls = append(toolCalls, map[string]any{
						"id":   p.ID,
						"type": "function",
						"function": map[string]any{
							"name":      p.Name,
							"arguments": string(p.Input),
						},
					})
				}
			}
			if text == "" && msg.Content.Text != "" {
				text = msg.Content.Text
			}

			am := map[string]any{
				"role":    "assistant",
				"content": text,
			}
			switch {
			case hasReasoning:
				am["reasoning_content"] = reasoning.String()
			case isV4:
				am["reasoning_content"] = ""
			}
			if len(toolCalls) > 0 {
				am["tool_calls"] = toolCalls
			}
			out = append(out, am)

		case message.RoleTool:
			for _, part := range msg.Content.Parts {
				tr, ok := part.(message.ToolResultPart)
				if !ok {
					continue
				}
				resultStr := ""
				switch v := tr.Result.(type) {
				case string:
					resultStr = v
				default:
					if b, err := json.Marshal(v); err == nil {
						resultStr = string(b)
					}
				}
				out = append(out, map[string]any{
					"role":         "tool",
					"content":      resultStr,
					"tool_call_id": tr.ToolCallID,
				})
			}
		}
	}
	return out, nil
}

// getText flattens message.Content to a string. Mirrors openaicompat's
// getTextFromContent (kept local to avoid an unexported cross-package
// dependency).
func getText(content message.Content) string {
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

// convertUserContent mirrors openaicompat's user-content shape: a bare
// string when the message has only text, otherwise a multipart array of
// {type: "text"|"image_url"} entries.
func convertUserContent(content message.Content) any {
	if content.Text != "" && len(content.Parts) == 0 {
		return content.Text
	}
	if len(content.Parts) == 1 {
		if tp, ok := content.Parts[0].(message.TextPart); ok {
			return tp.Text
		}
	}
	parts := make([]map[string]any, 0, len(content.Parts))
	for _, part := range content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			parts = append(parts, map[string]any{"type": "text", "text": p.Text})
		case message.ImagePart:
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": p.Image},
			})
		case message.FilePart:
			if strings.HasPrefix(p.MimeType, "text/") {
				decoded, err := base64.StdEncoding.DecodeString(p.Data)
				if err != nil {
					decoded = []byte(p.Data)
				}
				parts = append(parts, map[string]any{"type": "text", "text": string(decoded)})
			}
		}
	}
	return parts
}
