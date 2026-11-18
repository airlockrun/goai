package xai

import (
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

func (m *XaiResponsesModel) ValidateFile(role message.Role, file message.FilePart) error {
	if role != message.RoleTool && role != message.RoleUser {
		return provider.ErrUnsupportedFile
	}
	if !strings.HasPrefix(m.id, "grok-4") && !strings.HasPrefix(m.id, "grok-2-vision") && m.id != "grok-vision-beta" {
		return provider.ErrUnsupportedFile
	}
	mimes := []string{"image/png", "image/jpeg", "image/webp", "image/gif", "image/*"}
	if strings.HasPrefix(m.id, "grok-4") {
		mimes = append(mimes, "application/pdf")
	}
	return provider.ValidateInlineOrURLFile(file, mimes, true)
}

func (m *XaiResponsesModel) ValidateMessages(messages []message.Message) error {
	return provider.ValidateMessageContent(messages, m, false, nil)
}

func xaiFilePart(file message.FilePart) responsesContentPart {
	part := responsesContentPart{Type: "input_file"}
	image := strings.HasPrefix(file.MimeType, "image/")
	if image {
		part.Type = "input_image"
	}
	switch data := file.Data.(type) {
	case message.FileDataBytes:
		value := "data:" + file.MimeType + ";base64," + data.Data
		if image {
			part.ImageURL = value
		} else {
			part.FileData, part.MimeType, part.Filename = data.Data, file.MimeType, file.Filename
			if part.Filename == "" {
				part.Filename = "document.pdf"
			}
		}
	case message.FileDataURL:
		if image {
			part.ImageURL = data.URL
		} else {
			part.FileURL = data.URL
		}
	default:
		panic("xAI file builder requires validated data")
	}
	return part
}

func xaiToolOutput(output message.ToolResultOutput) any {
	if _, ok := output.(message.ContentOutput); !ok {
		return message.ToolOutputWire(output)
	}
	parts, err := message.ToolContentParts(output)
	if err != nil {
		panic("xAI tool builder requires validated content")
	}
	result := make([]responsesContentPart, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case message.TextPart:
			result = append(result, responsesContentPart{Type: "input_text", Text: p.Text})
		case message.FilePart:
			result = append(result, xaiFilePart(p))
		}
	}
	return result
}
