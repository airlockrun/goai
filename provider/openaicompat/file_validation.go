package openaicompat

import (
	"encoding/base64"
	"strings"
	"unicode/utf8"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

func (m *CompatModel) ValidateFile(role message.Role, file message.FilePart) error {
	if m.provider.opts.FileValidator != nil {
		if m.provider.opts.MessageConverter == nil {
			if err := validateChatFileShape(role, file); err != nil {
				return err
			}
		}
		return m.provider.opts.FileValidator(m.id, role, file)
	}
	return validateTextFile(role, file)
}

func (m *CompatModel) ValidateMessages(messages []message.Message) error {
	if err := m.validateMessageContent(messages); err != nil {
		return err
	}
	if convert := m.provider.opts.MessageConverter; convert != nil {
		_, err := convert(m.id, messages)
		return err
	}
	return nil
}

func (m *CompatModel) validateMessageContent(messages []message.Message) error {
	if err := provider.ValidateMessageContent(messages, m, true, nil); err != nil {
		return err
	}
	if m.provider.opts.MessageConverter == nil {
		return provider.ValidateTextToolMessages(messages)
	}
	return nil
}

// ValidateUserImageFile declares the user image_url contract for adapters that
// explicitly support it. It does not infer model vision or tool media support.
// Unknown generic endpoints must supply an explicit endpoint/model declaration.
func ValidateUserImageFile(_ string, role message.Role, file message.FilePart) error {
	if strings.HasPrefix(file.MimeType, "text/") {
		return validateTextFile(role, file)
	}
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, []string{"image/png", "image/jpeg", "image/gif", "image/webp", "image/*"}, true)
}

func validateTextFile(role message.Role, file message.FilePart) error {
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	if strings.HasPrefix(file.MimeType, "text/") {
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
	return provider.ErrUnsupportedFile
}

// validateChatFileShape checks encoding only. It is not an endpoint capability
// declaration; the instantiated model performs admission before conversion.
func validateChatFileShape(role message.Role, file message.FilePart) error {
	if strings.HasPrefix(file.MimeType, "text/") {
		return validateTextFile(role, file)
	}
	if role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	return provider.ValidateInlineOrURLFile(file, []string{"image/png", "image/jpeg", "image/gif", "image/webp", "image/*", "video/mp4", "video/webm", "video/mpeg", "video/quicktime"}, true)
}
