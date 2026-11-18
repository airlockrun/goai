package bedrock

import (
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/anthropic"
)

func (m *BedrockLanguageModel) ValidateFile(role message.Role, file message.FilePart) error {
	id := bedrockModelName(m.id)
	if strings.HasPrefix(id, "anthropic.") {
		return anthropic.ValidateFile(bedrockAnthropicConfig(m.id), role, file)
	}
	if role != message.RoleUser && role != message.RoleTool {
		return provider.ErrUnsupportedFile
	}
	// Nova Micro is text-only. Other families are not inferred multimodal
	// from the Converse API's broad content union.
	if !strings.HasPrefix(id, "amazon.nova-lite-v1") && !strings.HasPrefix(id, "amazon.nova-pro-v1") {
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, []string{"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"}, false)
}

func (m *BedrockLanguageModel) ValidateMessages(messages []message.Message) error {
	id := bedrockModelName(m.id)
	if strings.HasPrefix(id, "anthropic.") {
		return anthropic.ValidateMessages(bedrockAnthropicConfig(m.id), messages)
	}
	if strings.HasPrefix(id, "amazon.titan") || strings.HasPrefix(id, "meta.llama") || strings.HasPrefix(id, "mistral.") || strings.HasPrefix(id, "cohere.") {
		return provider.ValidatePromptMessages(messages)
	}
	return provider.ValidateMessageContent(messages, m, true, nil)
}

func converseFile(file message.FilePart) map[string]any {
	data := file.Data.(message.FileDataBytes)
	if file.MimeType == "application/pdf" {
		return map[string]any{"document": map[string]any{"format": "pdf", "name": "document", "source": map[string]any{"bytes": data.Data}}}
	}
	return map[string]any{"image": map[string]any{"format": strings.TrimPrefix(file.MimeType, "image/"), "source": map[string]any{"bytes": data.Data}}}
}

func converseToolOutput(output message.ToolResultOutput) []any {
	if _, ok := output.(message.ContentOutput); !ok {
		return []any{map[string]any{"text": message.ToolOutputText(output)}}
	}
	parts, err := message.ToolContentParts(output)
	if err != nil {
		panic("Converse tool builder requires validated content")
	}
	content := make([]any, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case message.TextPart:
			content = append(content, map[string]any{"text": p.Text})
		case message.FilePart:
			content = append(content, converseFile(p))
		}
	}
	return content
}
