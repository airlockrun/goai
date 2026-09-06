package mistral

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider/openaicompat"
)

func convertMessages(_ string, messages []message.Message) ([]any, error) {
	var out []any
	for i, msg := range messages {
		converted, err := openaicompat.ConvertMessages([]message.Message{msg})
		if err != nil {
			return nil, err
		}
		if msg.Role == message.RoleAssistant {
			body := converted[0].(map[string]any)
			var parts []any
			hasReasoning := false
			if msg.Content.Text != "" {
				parts = append(parts, map[string]any{"type": "text", "text": msg.Content.Text})
			}
			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.TextPart:
					parts = append(parts, map[string]any{"type": "text", "text": p.Text})
				case message.ReasoningPart:
					hasReasoning = true
					parts = append(parts, map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": p.Text}}, "closed": true})
				}
			}
			if hasReasoning {
				body["content"] = parts
			}
			if i == len(messages)-1 {
				body["prefix"] = true
			}
		}
		out = append(out, converted...)
	}
	return out, nil
}
