package cohere

import (
	"encoding/base64"
	"errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider/openaicompat"
	"strings"
)

func convertMessages(messages []message.Message) ([]any, []any, error) {
	var out, documents []any
	for _, msg := range messages {
		if msg.Role == message.RoleUser {
			var parts []message.Part
			for _, part := range msg.Content.Parts {
				file, ok := part.(message.FilePart)
				if !ok || strings.HasPrefix(file.MimeType, "image/") {
					parts = append(parts, part)
					continue
				}
				var text string
				switch data := file.Data.(type) {
				case message.FileDataText:
					text = data.Text
				case message.FileDataBytes:
					decoded, err := base64.StdEncoding.DecodeString(data.Data)
					if err != nil {
						return nil, nil, err
					}
					text = string(decoded)
				default:
					return nil, nil, errors.New("Cohere documents require inline file data")
				}
				data := map[string]any{"text": text}
				if file.Filename != "" {
					data["title"] = file.Filename
				}
				documents = append(documents, map[string]any{"data": data})
			}
			msg.Content.Parts = parts
		}
		converted, err := openaicompat.ConvertMessages([]message.Message{msg})
		if err != nil {
			return nil, nil, err
		}
		if msg.Role == message.RoleAssistant {
			body := converted[0].(map[string]any)
			if _, ok := body["tool_calls"]; ok {
				delete(body, "content")
			}
		}
		out = append(out, converted...)
	}
	return out, documents, nil
}
