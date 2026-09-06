package cerebras

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequestNormalization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["max_completion_tokens"] != float64(12) || body["max_tokens"] != nil {
			t.Errorf("body = %v", body)
		}
		msg := body["messages"].([]any)[0].(map[string]any)
		if msg["reasoning"] != "think" || msg["reasoning_content"] != nil {
			t.Errorf("message = %v", msg)
		}
		w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer server.Close()
	max := 12
	events, err := New(Options{BaseURL: server.URL}).Model("gpt-oss-120b").Stream(context.Background(), &stream.CallOptions{MaxOutputTokens: &max, Messages: []message.Message{{Role: message.RoleAssistant, Content: message.Content{Parts: []message.Part{message.ReasoningPart{Text: "think"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	for event := range events {
		if event.Type == stream.EventError {
			t.Fatal(event.Data)
		}
	}
}
