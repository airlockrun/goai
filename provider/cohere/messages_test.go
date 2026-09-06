package cohere

import (
	"encoding/json"
	"github.com/airlockrun/goai/message"
	"testing"
)

func TestDocumentsAndAssistantTools(t *testing.T) {
	input := []message.Message{
		{Role: message.RoleUser, Content: message.Content{Parts: []message.Part{message.TextPart{Text: "read"}, message.FilePart{MimeType: "text/plain", Filename: "note.txt", Data: message.FileDataBytes{Data: "bm90ZQ=="}}}}},
		{Role: message.RoleAssistant, Content: message.Content{Parts: []message.Part{message.ToolCallPart{ID: "call", Name: "tool", Input: json.RawMessage(`{}`)}}}},
	}
	output, documents, err := convertMessages(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(documents) != 1 || documents[0].(map[string]any)["data"].(map[string]any)["text"] != "note" {
		t.Fatalf("documents = %v", documents)
	}
	if _, ok := output[1].(map[string]any)["content"]; ok {
		t.Fatal("assistant tool turn contains content")
	}
	if len(input[0].Content.Parts) != 2 {
		t.Fatal("input mutated")
	}
}
