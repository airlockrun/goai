package message

import (
	"errors"
	"slices"
	"strings"
)

// ToolContentParts normalizes multipart tool output without changing persisted
// messages. Non-content outputs return nil; unknown variants return a static
// error. File bytes remain base64 and are never rendered as text.
func ToolContentParts(output ToolResultOutput) ([]Part, error) {
	content, ok := output.(ContentOutput)
	if !ok {
		return nil, nil
	}
	parts := make([]Part, 0, len(content.Value))
	for _, item := range content.Value {
		if item.Type == "text" {
			parts = append(parts, TextPart{Text: item.Text, ProviderOptions: item.ProviderOptions})
			continue
		}
		file := FilePart{MimeType: item.MediaType, Filename: item.Filename, ProviderOptions: item.ProviderOptions}
		switch item.Type {
		case "image-data", "file-data":
			file.Data = FileDataBytes{Data: item.Data}
		case "image-url", "file-url":
			file.Data = FileDataURL{URL: item.URL}
			// An image URL identifies the media category without a fetched MIME.
			if item.Type == "image-url" && file.MimeType == "" {
				file.MimeType = "image/*"
			}
		default:
			return nil, errors.New("unsupported tool content representation")
		}
		if (item.Type == "image-data" || item.Type == "image-url") &&
			(len(file.MimeType) < 6 || file.MimeType[:6] != "image/") {
			return nil, errors.New("invalid tool image representation")
		}
		if data, ok := file.Data.(FileDataURL); ok && strings.HasPrefix(data.URL, "data:") {
			header, encoded, ok := strings.Cut(strings.TrimPrefix(data.URL, "data:"), ",")
			mime, base64 := strings.CutSuffix(header, ";base64")
			if !ok || !base64 || mime == "" || (file.MimeType != mime && file.MimeType != "image/*") || (file.MimeType == "image/*" && !strings.HasPrefix(mime, "image/")) {
				return nil, errors.New("invalid inline file URL representation")
			}
			file.MimeType, file.Data = mime, FileDataBytes{Data: encoded}
		}
		parts = append(parts, file)
	}
	return parts, nil
}

// ToolMessageResults assigns sibling text/files to the final actual result.
// Returned result/output slices are request-local copies. Attachments retain
// their sibling order after that result's own output; no result is synthesized.
func ToolMessageResults(msg Message) ([]ToolResultPart, error) {
	if msg.Role != RoleTool || msg.Content.Text != "" {
		return nil, errors.New("unsupported tool message representation")
	}
	var results []ToolResultPart
	var attachments []ToolContentItem
	for _, part := range msg.Content.Parts {
		switch p := part.(type) {
		case ToolResultPart:
			if output, ok := p.Output.(ContentOutput); ok {
				output.Value = slices.Clone(output.Value)
				p.Output = output
			}
			results = append(results, p)
		case TextPart:
			attachments = append(attachments, ToolContentItem{Type: "text", Text: p.Text, ProviderOptions: p.ProviderOptions})
		case FilePart:
			item := ToolContentItem{MediaType: p.MimeType, Filename: p.Filename, ProviderOptions: p.ProviderOptions}
			switch data := p.Data.(type) {
			case FileDataBytes:
				item.Type, item.Data = "file-data", data.Data
			case FileDataURL:
				item.Type, item.URL = "file-url", data.URL
			default:
				return nil, errors.New("unsupported tool attachment representation")
			}
			attachments = append(attachments, item)
		default:
			return nil, errors.New("unsupported tool message representation")
		}
	}
	if len(attachments) == 0 {
		return results, nil
	}
	if len(results) == 0 {
		return nil, errors.New("tool attachments require a tool result")
	}
	last := &results[len(results)-1]
	var items []ToolContentItem
	if content, ok := last.Output.(ContentOutput); ok {
		items = append(items, content.Value...)
	} else {
		items = append(items, ToolContentItem{Type: "text", Text: ToolOutputWire(last.Output)})
	}
	last.Output = ContentOutput{Value: append(items, attachments...)}
	return results, nil
}
