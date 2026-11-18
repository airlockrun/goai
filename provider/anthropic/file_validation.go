package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"path"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

func (m *AnthropicModel) ValidateFile(role message.Role, file message.FilePart) error {
	return ValidateFile(m.provider.cfg, role, file)
}

func (m *AnthropicModel) ValidateMessages(messages []message.Message) error {
	return ValidateMessages(m.provider.cfg, messages)
}

// ValidateMessages admits native hosted-result replay only when its actual
// assistant raw-block representation is present. Generic multipart outputs do
// not stand in for a hosted result block.
func ValidateMessages(cfg Config, messages []message.Message) error {
	validator := provider.FileValidationFunc(func(role message.Role, file message.FilePart) error { return ValidateFile(cfg, role, file) })
	return provider.ValidateMessageContent(messages, validator, true, func(result message.ToolResultPart) bool {
		if _, ok := result.Output.(message.ContentOutput); ok {
			return false
		}
		opts, _ := result.ProviderOptions["anthropic"].(map[string]any)
		encoded, err := json.Marshal(opts["rawBlock"])
		if err != nil {
			return false
		}
		var raw map[string]any
		if json.Unmarshal(encoded, &raw) != nil {
			return false
		}
		kind, _ := raw["type"].(string)
		if !strings.HasSuffix(kind, "_tool_result") || raw["tool_use_id"] != result.ToolCallID {
			return false
		}
		return true
	})
}

// ValidateFile applies the configured Messages transport's source restrictions.
// Vertex and Bedrock URLs must be materialized by the caller, not fetched here.
func ValidateFile(cfg Config, role message.Role, file message.FilePart) error {
	if role != message.RoleUser && role != message.RoleTool {
		return provider.ErrUnsupportedFile
	}
	mimes := []string{"image/png", "image/jpeg", "image/gif", "image/webp", "image/*", "application/pdf"}
	if file.MimeType == "text/plain" {
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
	if err := provider.ValidateInlineOrURLFile(file, mimes, true); err != nil {
		return err
	}
	if data, ok := file.Data.(message.FileDataURL); ok {
		if cfg.ProviderID == "amazon-bedrock" || cfg.ProviderID == "vertex-anthropic" {
			return provider.ErrUnsupportedFile
		}
		if cfg.SupportedURLs != nil {
			for pattern, urls := range cfg.SupportedURLs() {
				matched, _ := path.Match(pattern, file.MimeType)
				if !matched {
					continue
				}
				for _, pattern := range urls {
					matched, err := regexp.MatchString(pattern, data.URL)
					if err == nil && matched {
						return nil
					}
				}
			}
			return provider.ErrUnsupportedFile
		}
	}
	return nil
}
