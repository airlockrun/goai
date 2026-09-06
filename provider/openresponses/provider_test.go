package openresponses

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func TestProvider_Responses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("X-Custom") != "call" {
			t.Errorf("headers: %v", r.Header)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		if req["model"] != "gpt-6" || req["temperature"] != 0.5 {
			t.Errorf("generic request: %v", req)
		}
		if req["input"].([]any)[0].(map[string]any)["role"] != "system" {
			t.Error("generic endpoint must not infer OpenAI roles")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: " + `{"type":"response.output_text.delta","delta":"hello"}` + "\n\ndata: " + `{"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":1}}}` + "\n\n"))
	}))
	defer server.Close()
	p := New(Options{BaseURL: server.URL + "/v1/", APIKey: "key", Headers: map[string]string{"X-Custom": "provider"}})
	m := p.Model("gpt-6")
	if m.Provider() != "openresponses.responses" {
		t.Fatal(m.Provider())
	}
	temperature := 0.5
	events, err := m.Stream(context.Background(), &stream.CallOptions{Messages: []message.Message{message.NewSystemMessage("rules")}, Temperature: &temperature, Headers: map[string]string{"X-Custom": "call", "Authorization": "untrusted"}})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var finished bool
	for event := range events {
		switch event.Type {
		case stream.EventError:
			t.Fatal(event.Data)
		case stream.EventTextDelta:
			text += event.Data.(stream.TextDeltaEvent).Text
		case stream.EventFinish:
			finished = true
		}
	}
	if text != "hello" || !finished {
		t.Fatalf("text=%q finished=%v", text, finished)
	}
}

func TestProvider_RequiresEndpoint(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	New(Options{})
}
