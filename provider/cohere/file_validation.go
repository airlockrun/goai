package cohere

import (
	"encoding/base64"
	"strings"
	"unicode/utf8"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

func (m *CohereModel) ValidateFile(role message.Role, file message.FilePart) error {
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	if strings.HasPrefix(file.MimeType, "text/") || file.MimeType == "application/json" {
		switch data := file.Data.(type) {
		case message.FileDataText:
			if utf8.ValidString(data.Text) {
				return nil
			}
		case message.FileDataBytes:
			decoded, err := base64.StdEncoding.DecodeString(data.Data)
			if err == nil && utf8.Valid(decoded) {
				return nil
			}
		}
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, []string{"image/png", "image/jpeg", "image/gif", "image/webp", "image/*"}, true)
}

func (m *CohereModel) ValidateMessages(messages []message.Message) error {
	if err := provider.ValidateMessageContent(messages, m, false, nil); err != nil {
		return err
	}
	return provider.ValidateTextToolMessages(messages)
}
