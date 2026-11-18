package openai

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

var responseFileMIMEs = []string{"image/png", "image/jpeg", "image/webp", "image/gif", "image/*", "application/pdf", "text/plain", "text/markdown", "text/csv", "text/html", "application/json", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

func (m *ResponsesModel) ValidateFile(role message.Role, file message.FilePart) error {
	if role != message.RoleUser && role != message.RoleTool {
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, responseFileMIMEs, true)
}

func (m *ResponsesModel) ValidateMessages(messages []message.Message) error {
	return provider.ValidateMessageContent(messages, m, false, nil)
}

func (m *ChatModel) ValidateMessages(messages []message.Message) error {
	if err := provider.ValidateMessageContent(messages, m, false, nil); err != nil {
		return err
	}
	return provider.ValidateTextToolMessages(messages)
}

func (m *ChatModel) ValidateFile(role message.Role, file message.FilePart) error {
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	if file.MimeType == "application/pdf" || file.MimeType == "audio/wav" || file.MimeType == "audio/mpeg" || file.MimeType == "audio/mp3" {
		return provider.ValidateInlineOrURLFile(file, []string{file.MimeType}, false)
	}
	return provider.ValidateInlineOrURLFile(file, responseFileMIMEs[:5], true)
}
