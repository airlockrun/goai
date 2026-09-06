package vertex

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/stream"
	"strings"
	"testing"
)

func TestPartialArgs(t *testing.T) {
	events := make(chan stream.Event, 100)
	data := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","partialArgs":[{"jsonPath":"$.cities[0]","stringValue":"New ","willContinue":true}],"willContinue":true}}]}}]}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.cities[0]","stringValue":"York"},{"jsonPath":"$.count","numberValue":1}]}}]},"finishReason":"STOP"}]}

`
	(&VertexLanguageModel{}).processStream(context.Background(), strings.NewReader(data), nil, events, false)
	close(events)
	calls := 0
	for event := range events {
		if event.Type == stream.EventError {
			t.Fatalf("error = %+v", event.Data)
		}
		if call, ok := event.Data.(stream.ToolCallEvent); ok {
			calls++
			var args map[string]any
			if err := json.Unmarshal(call.Input, &args); err != nil {
				t.Fatal(err)
			}
			if args["cities"].([]any)[0] != "New York" || args["count"] != float64(1) {
				t.Fatalf("args = %v", args)
			}
		}
	}
	if calls != 1 {
		t.Fatalf("calls = %d", calls)
	}
}
