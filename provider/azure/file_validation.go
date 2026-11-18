package azure

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

func (m *AzureLanguageModel) ValidateFile(role message.Role, file message.FilePart) error {
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, []string{"image/png", "image/jpeg", "image/gif", "image/webp", "image/*"}, true)
}

func (m *AzureLanguageModel) ValidateMessages(messages []message.Message) error {
	if err := provider.ValidateMessageContent(messages, m, false, nil); err != nil {
		return err
	}
	return provider.ValidateTextToolMessages(messages)
}
