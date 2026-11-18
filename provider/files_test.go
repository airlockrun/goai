package provider

import (
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
)

func TestValidateInlineOrURLFile(t *testing.T) {
	for _, test := range []struct {
		name, mime string
		data       message.FileData
		url, want  bool
	}{
		{"inline", "image/png", message.FileDataBytes{Data: "c3BlY2ltZW4="}, false, true},
		{"url", "image/png", message.FileDataURL{URL: "https://example.invalid/image.png"}, true, true},
		{"url-disabled", "image/png", message.FileDataURL{URL: "https://example.invalid/image.png"}, false, false},
		{"missing-mime", "", message.FileDataBytes{Data: "PRIVATE"}, true, false},
		{"unsupported-mime", "application/PRIVATE", message.FileDataBytes{Data: "PRIVATE"}, true, false},
		{"missing-bytes", "image/png", message.FileDataBytes{}, true, false},
		{"wildcard-bytes", "image/*", message.FileDataBytes{Data: "PRIVATE"}, true, false},
		{"wildcard-url", "image/*", message.FileDataURL{URL: "https://example.invalid/image"}, true, true},
		{"provider-reference", "image/png", message.FileDataReference{Reference: map[string]any{"PRIVATE": "PRIVATE"}}, true, false},
		{"inline-text", "image/png", message.FileDataText{Text: "PRIVATE"}, true, false},
		{"internal-reference", "image/png", message.FileDataURL{URL: "airlock-file:PRIVATE"}, true, false},
		{"credentials", "image/png", message.FileDataURL{URL: "https://PRIVATE:PRIVATE@example.invalid/image"}, true, false},
		{"nil-data", "image/png", nil, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateInlineOrURLFile(message.FilePart{MimeType: test.mime, Data: test.data}, []string{"image/png", "image/*"}, test.url)
			if (err == nil) != test.want {
				t.Fatalf("support mismatch: %v", err)
			}
			if err != nil && strings.Contains(err.Error(), "PRIVATE") {
				t.Fatal("diagnostic leaked input")
			}
		})
	}
}

func TestValidateMessageFilesRejectsOrphanAndUnknownContent(t *testing.T) {
	validator := FileValidationFunc(func(message.Role, message.FilePart) error { return nil })
	for _, msg := range []message.Message{
		{Role: message.RoleTool, Content: message.Content{Parts: []message.Part{message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "PRIVATE"}}}}},
		{Role: message.RoleTool, Content: message.Content{Text: "PRIVATE"}},
		message.NewToolMessage("PRIVATE", "PRIVATE", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "PRIVATE"}}}),
	} {
		if err := ValidateMessageFiles([]message.Message{msg}, validator); err == nil || strings.Contains(err.Error(), "PRIVATE") {
			t.Fatalf("expected static rejection: %v", err)
		}
	}
}
