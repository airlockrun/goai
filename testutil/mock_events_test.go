package testutil

import (
	"errors"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/stream"
)

func TestMockEventHelpers(t *testing.T) {
	usage := MockUsage(5, 10)
	for _, tt := range []struct {
		name   string
		events []stream.Event
		types  []stream.EventType
		reason stream.FinishReason
	}{
		{"text", MockTextResponse("Hello!", usage), []stream.EventType{stream.EventTextStart, stream.EventTextDelta, stream.EventTextEnd, stream.EventFinish}, stream.FinishReasonStop},
		{"chunks", MockStreamedTextResponse([]string{"Hello", "!"}, usage), []stream.EventType{stream.EventTextStart, stream.EventTextDelta, stream.EventTextDelta, stream.EventTextEnd, stream.EventFinish}, stream.FinishReasonStop},
		{"tool", MockToolCallResponse("call_123", "get_weather", map[string]string{"location": "NYC"}, usage), []stream.EventType{stream.EventToolCall, stream.EventFinish}, stream.FinishReasonToolCalls},
		{"text and tool", MockTextWithToolCallResponse("Hello!", "call_123", "get_weather", map[string]string{"location": "NYC"}, usage), []stream.EventType{stream.EventTextStart, stream.EventTextDelta, stream.EventTextEnd, stream.EventToolCall, stream.EventFinish}, stream.FinishReasonToolCalls},
		{"reasoning", MockReasoningResponse("thinking", "Hello!", usage), []stream.EventType{stream.EventReasoningStart, stream.EventReasoningDelta, stream.EventReasoningEnd, stream.EventTextStart, stream.EventTextDelta, stream.EventTextEnd, stream.EventFinish}, stream.FinishReasonStop},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var types []stream.EventType
			var text string
			for _, event := range tt.events {
				types = append(types, event.Type)
				switch data := event.Data.(type) {
				case stream.TextDeltaEvent:
					text += data.Text
				case stream.ToolCallEvent:
					if data.ToolCallID != "call_123" || data.ToolName != "get_weather" || string(data.Input) != `{"location":"NYC"}` {
						t.Fatalf("tool call = %+v", data)
					}
				case stream.ReasoningDeltaEvent:
					if data.ID != "r0" || data.Text != "thinking" {
						t.Fatalf("reasoning = %+v", data)
					}
				}
			}
			if !reflect.DeepEqual(types, tt.types) {
				t.Fatalf("types = %v, want %v", types, tt.types)
			}
			if tt.name != "tool" && text != "Hello!" {
				t.Fatalf("text = %q", text)
			}
			finish := tt.events[len(tt.events)-1].Data.(stream.FinishEvent)
			if finish.FinishReason != tt.reason || finish.Usage.InputTotal() != 5 || finish.Usage.OutputTotal() != 10 || finish.Usage.GrandTotal() != 15 {
				t.Fatalf("finish = %+v", finish)
			}
		})
	}
}

func TestMockErrorResponse(t *testing.T) {
	err := errors.New("test error")
	events := MockErrorResponse(err)
	if len(events) != 1 || events[0].Type != stream.EventError || events[0].Data.(stream.ErrorEvent).Error != err {
		t.Fatalf("events = %+v", events)
	}
}

func TestMockToolCallResponseRejectsInvalidInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("invalid input accepted")
		}
	}()
	MockToolCallResponse("c", "tool", make(chan int), stream.Usage{})
}
