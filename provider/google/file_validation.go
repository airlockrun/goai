package google

import (
	"net/url"
	"path"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
)

// SupportsFunctionResponseParts selects generateContent's native multimodal
// function response contract. Unknown model families are not inferred capable.
func SupportsFunctionResponseParts(modelID string) bool {
	id := strings.ToLower(path.Base(modelID))
	return strings.HasPrefix(id, "gemini-3-") || strings.HasPrefix(id, "gemini-3.")
}

// ValidateFile validates the shared Gemini generateContent file transport.
// Native Gemini 3 function responses accept confirmed image/PDF/audio/video
// MIME types. Older-model sibling responses are limited to images and PDFs.
func ValidateFile(modelID string, role message.Role, file message.FilePart, vertex bool) error {
	id := strings.ToLower(path.Base(modelID))
	known := SupportsFunctionResponseParts(id) || strings.HasPrefix(id, "gemini-2.5-") || strings.HasPrefix(id, "gemini-2.0-") || strings.HasPrefix(id, "gemini-1.5-")
	if !known || (role != message.RoleUser && role != message.RoleTool) {
		return provider.ErrUnsupportedFile
	}
	mimes := []string{"image/png", "image/jpeg", "image/webp", "application/pdf"}
	if role == message.RoleTool && SupportsFunctionResponseParts(modelID) {
		mimes = append(mimes, "audio/wav", "audio/mp3", "audio/mpeg", "audio/aac", "audio/ogg", "audio/flac", "video/mp4", "video/mpeg", "video/webm")
	}
	if role == message.RoleUser {
		mimes = append(mimes, "image/heic", "image/heif", "audio/wav", "audio/mpeg", "audio/mp3", "audio/aac", "audio/ogg", "audio/flac", "audio/mp4", "video/mp4", "video/mpeg", "video/quicktime", "video/webm")
	}
	if data, ok := file.Data.(message.FileDataURL); ok {
		// Google Files API and Vertex GCS use provider-specific URI sources.
		u, err := url.Parse(data.URL)
		if err != nil {
			return provider.ErrUnsupportedFile
		}
		valid := u.Scheme == "https" && u.Host == "generativelanguage.googleapis.com" && strings.HasPrefix(u.Path, "/v1beta/files/")
		if vertex {
			valid = u.Scheme == "gs" && u.Host != "" && u.Path != ""
		}
		if !valid || u.User != nil {
			return provider.ErrUnsupportedFile
		}
		copy := file
		copy.Data = message.FileDataBytes{Data: "source-validation"}
		return provider.ValidateInlineOrURLFile(copy, mimes, false)
	}
	return provider.ValidateInlineOrURLFile(file, mimes, false)
}

func (m *GoogleModel) ValidateFile(role message.Role, file message.FilePart) error {
	return ValidateFile(m.id, role, file, false)
}

func (m *GoogleModel) ValidateMessages(messages []message.Message) error {
	return provider.ValidateMessageContent(messages, m, true, nil)
}

// BuildToolResponse preserves tool correlation and ordered media. Gemini 3
// uses native functionResponse.parts. Confirmed older generateContent models
// use sibling media in the same function-response user turn, not a new turn.
// Callers validate the complete message before invoking this builder.
func BuildToolResponse(modelID string, result message.ToolResultPart) (map[string]any, []message.FilePart, error) {
	response := map[string]any{"name": result.ToolName}
	if result.ToolCallID != "" {
		response["id"] = result.ToolCallID
	}
	content, ok := result.Output.(message.ContentOutput)
	if !ok {
		response["response"] = map[string]any{"result": message.ToolOutputWire(result.Output)}
		return response, nil, nil
	}
	parts, err := message.ToolContentParts(content)
	if err != nil {
		return nil, nil, err
	}
	var text strings.Builder
	var files []message.FilePart
	var media []map[string]any
	for _, part := range parts {
		switch p := part.(type) {
		case message.TextPart:
			text.WriteString(p.Text)
		case message.FilePart:
			files = append(files, p)
			media = append(media, FileWirePart(p))
		}
	}
	response["response"] = map[string]any{"result": text.String()}
	if SupportsFunctionResponseParts(modelID) {
		if len(media) > 0 {
			response["parts"] = media
		}
		return response, nil, nil
	}
	return response, files, nil
}

// FileWirePart returns the typed Gemini wire shape of a validated file.
func FileWirePart(file message.FilePart) map[string]any {
	data := map[string]any{"mimeType": file.MimeType}
	if file.Filename != "" {
		data["displayName"] = file.Filename
	}
	switch source := file.Data.(type) {
	case message.FileDataBytes:
		data["data"] = source.Data
		return map[string]any{"inlineData": data}
	case message.FileDataURL:
		data["fileUri"] = source.URL
		return map[string]any{"fileData": data}
	default:
		panic("Gemini file builder requires validated data")
	}
}
