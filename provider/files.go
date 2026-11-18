package provider

import (
	"errors"
	"net/url"
	"slices"

	"github.com/airlockrun/goai/message"
)

// FileValidator is an adapter's role-aware transport contract. Hosts must also
// enforce selected-model modalities, authorization and resource limits. An
// adapter without this interface does not declare file support. Complete-message
// content admission additionally requires MessageValidator.
type FileValidator interface {
	ValidateFile(message.Role, message.FilePart) error
}

// MessageValidator is the authoritative complete-message admission contract.
// FileValidator and ValidateMessageFiles alone cannot establish whether a
// serializer supports a tool result's placement or sibling representation.
type MessageValidator interface {
	ValidateMessages([]message.Message) error
}

// ValidateMessageContent combines file checks with actual serializer placement
// rules. hostedReplay is nil for serializers without assistant hosted replay.
// A non-nil declaration must verify the provider's replay representation.
func ValidateMessageContent(messages []message.Message, validator FileValidator, siblingText bool, hostedReplay func(message.ToolResultPart) bool) error {
	if err := ValidateMessageFiles(messages, validator); err != nil {
		return err
	}
	for _, msg := range messages {
		for _, part := range msg.Content.Parts {
			if result, ok := part.(message.ToolResultPart); ok && msg.Role != message.RoleTool {
				if msg.Role != message.RoleAssistant || !result.ProviderExecuted || hostedReplay == nil || !hostedReplay(result) {
					return ErrUnsupportedFile
				}
			}
			if msg.Role == message.RoleTool && !siblingText {
				if _, ok := part.(message.TextPart); ok {
					return ErrUnsupportedFile
				}
			}
		}
	}
	return nil
}

// FileValidationFunc adapts a validation function to FileValidator.
type FileValidationFunc func(message.Role, message.FilePart) error

func (f FileValidationFunc) ValidateFile(role message.Role, file message.FilePart) error {
	return f(role, file)
}

// ErrUnsupportedFile is intentionally independent of payload, MIME, URL and
// tool identity so it is safe to display or log.
var ErrUnsupportedFile = errors.New("file representation is not supported by this adapter for this role and model")

// ValidateInlineOrURLFile checks an exact MIME allowlist and source shape.
// image/* is accepted only for URLs when explicitly present in the allowlist.
// It does not fetch, decode, rewrite, or mutate data.
func ValidateInlineOrURLFile(file message.FilePart, mimes []string, allowURL bool) error {
	if !slices.Contains(mimes, file.MimeType) {
		return ErrUnsupportedFile
	}
	switch data := file.Data.(type) {
	case message.FileDataBytes:
		if data.Data == "" || file.MimeType == "image/*" {
			return ErrUnsupportedFile
		}
	case message.FileDataURL:
		u, err := url.Parse(data.URL)
		if !allowURL || err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || u.User != nil {
			return ErrUnsupportedFile
		}
	default:
		return ErrUnsupportedFile
	}
	return nil
}

// ValidateMessageFiles checks file shapes, not serializer placement admission.
// Hosts must additionally use the actual adapter's MessageValidator.
func ValidateMessageFiles(messages []message.Message, validator FileValidator) error {
	for _, msg := range messages {
		if msg.Role == message.RoleTool {
			if msg.Content.Text != "" {
				return ErrUnsupportedFile
			}
			hasResult, hasAttachment := false, false
			for _, part := range msg.Content.Parts {
				switch part.(type) {
				case message.ToolResultPart:
					hasResult = true
				case message.FilePart, message.TextPart:
					hasAttachment = true
				default:
					return ErrUnsupportedFile
				}
			}
			if hasAttachment && !hasResult {
				return ErrUnsupportedFile
			}
		}
		for _, part := range msg.Content.Parts {
			if file, ok := part.(message.FilePart); ok {
				if err := validator.ValidateFile(msg.Role, file); err != nil {
					return err
				}
			}
			if result, ok := part.(message.ToolResultPart); ok {
				if msg.Role != message.RoleTool && !result.ProviderExecuted {
					return ErrUnsupportedFile
				}
				parts, err := message.ToolContentParts(result.Output)
				if err != nil {
					return err
				}
				for _, nested := range parts {
					if file, ok := nested.(message.FilePart); ok {
						if err := validator.ValidateFile(message.RoleTool, file); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// ValidateTextToolMessages rejects all non-text tool content without imposing
// a user-role file contract on provider-specific converters.
func ValidateTextToolMessages(messages []message.Message) error {
	for _, msg := range messages {
		if msg.Role != message.RoleTool {
			continue
		}
		if msg.Content.Text != "" {
			return ErrUnsupportedFile
		}
		for _, part := range msg.Content.Parts {
			result, ok := part.(message.ToolResultPart)
			if !ok {
				return ErrUnsupportedFile
			}
			parts, err := message.ToolContentParts(result.Output)
			if err != nil {
				return err
			}
			for _, nested := range parts {
				if _, ok := nested.(message.TextPart); !ok {
					return ErrUnsupportedFile
				}
			}
		}
	}
	return nil
}

// ValidatePromptMessages rejects structured input that prompt-only builders
// cannot represent. Plain Content.Text remains supported.
func ValidatePromptMessages(messages []message.Message) error {
	for _, msg := range messages {
		if msg.Role == message.RoleTool || len(msg.Content.Parts) != 0 {
			return errors.New("structured messages are not supported by this prompt adapter")
		}
	}
	return nil
}
