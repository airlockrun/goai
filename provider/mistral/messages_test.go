package mistral

import (
	"github.com/airlockrun/goai/message"
	"testing"
)

func TestReasoningReplay(t *testing.T) {
	messages := []message.Message{{Role: message.RoleAssistant, Content: message.Content{Parts: []message.Part{message.ReasoningPart{Text: "think"}, message.TextPart{Text: "answer"}}}}}
	out, err := convertMessages("mistral", messages)
	if err != nil {
		t.Fatal(err)
	}
	body := out[0].(map[string]any)
	parts := body["content"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["type"] != "thinking" || parts[0].(map[string]any)["closed"] != true || body["prefix"] != true {
		t.Fatalf("body = %v", body)
	}
}
