package cohere

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestV2ChatRequestAndToolStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/chat" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if _, ok := body["message"]; ok {
			t.Error("unexpected singular message")
		}
		if len(body["messages"].([]any)) != 2 || body["tool_choice"] != "REQUIRED" {
			t.Errorf("body = %v", body)
		}
		function := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
		if function["parameters"].(map[string]any)["additionalProperties"] != false {
			t.Error("tool schema was not preserved")
		}
		for _, chunk := range []string{
			`{"type":"content-start","delta":{"message":{"content":{"type":"thinking","thinking":"plan"}}}}`,
			`{"type":"content-end"}`,
			`{"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"call-a","function":{"name":"lookup","arguments":"{"}}}}}`,
			`{"type":"tool-call-delta","delta":{"message":{"tool_calls":{"function":{"arguments":"\"x\":1}"}}}}}`,
			`{"type":"tool-call-end"}`,
			`{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":4,"output_tokens":2},"cached_tokens":1,"custom":5}}}`,
		} {
			w.Write([]byte("data: " + chunk + "\n\n"))
		}
	}))
	defer server.Close()
	model := New(Options{BaseURL: server.URL + "/v1"}).Model("command-a")
	events, err := model.Stream(context.Background(), &stream.CallOptions{Messages: []message.Message{message.NewSystemMessage("system"), message.NewUserMessage("query")}, Tools: []tool.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object","additionalProperties":false}`)}}, ToolChoice: "required"})
	if err != nil {
		t.Fatal(err)
	}
	var call stream.ToolCallEvent
	var reasoning string
	var finish stream.FinishEvent
	for event := range events {
		switch event.Type {
		case stream.EventError:
			t.Fatal(event.Data.(stream.ErrorEvent).Error)
		case stream.EventToolCall:
			call = event.Data.(stream.ToolCallEvent)
		case stream.EventReasoningDelta:
			reasoning += event.Data.(stream.ReasoningDeltaEvent).Text
		case stream.EventFinish:
			finish = event.Data.(stream.FinishEvent)
		}
	}
	if call.ToolCallID != "call-a" || string(call.Input) != `{"x":1}` || reasoning != "plan" || finish.FinishReason != stream.FinishReasonToolCalls || finish.Usage.Raw["custom"] != float64(5) {
		t.Fatalf("call=%+v reasoning=%q finish=%+v", call, reasoning, finish)
	}
}

func TestV2IncompleteStream(t *testing.T) {
	m := New(Options{}).Model("command-a").(*CohereModel)
	for _, input := range []string{"data: {}\n\n", "data: {bad}\n\n"} {
		t.Run(input, func(t *testing.T) {
			events := make(chan stream.Event, 20)
			m.processStream(context.Background(), strings.NewReader(input), nil, events, false)
			close(events)
			failed := false
			for event := range events {
				if event.Type == stream.EventError {
					failed = true
				}
				if event.Type == stream.EventFinish {
					t.Fatal("unexpected finish")
				}
			}
			if !failed {
				t.Fatal("expected error")
			}
		})
	}
}
