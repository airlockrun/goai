package cerebras

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider/openaicompat"
	"strings"
)

func convertMessages(_ string, messages []message.Message) ([]any, error) {
	var out []any
	for _, msg := range messages {
		converted, err := openaicompat.ConvertMessages([]message.Message{msg})
		if err != nil {
			return nil, err
		}
		if msg.Role == message.RoleAssistant {
			var reasoning strings.Builder
			for _, part := range msg.Content.Parts {
				if p, ok := part.(message.ReasoningPart); ok {
					reasoning.WriteString(p.Text)
				}
			}
			if reasoning.Len() > 0 {
				converted[0].(map[string]any)["reasoning"] = reasoning.String()
			}
		}
		out = append(out, converted...)
	}
	return out, nil
}
