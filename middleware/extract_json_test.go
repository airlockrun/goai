package middleware

import (
	"strings"
	"testing"

	"github.com/airlockrun/goai/stream"
)

// collectText concatenates all TextDeltaEvent.Text from an event slice.
func collectText(events []stream.Event) string {
	var sb strings.Builder
	for _, ev := range events {
		if d, ok := ev.Data.(stream.TextDeltaEvent); ok {
			sb.WriteString(d.Text)
		}
	}
	return sb.String()
}

func TestExtractJson_StripsDefaultMarkdownFence(t *testing.T) {
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "```json\n"}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: `{"answer":"42"}`}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "\n```"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop}},
	})
	got := drainAll(t, WrapModel(inner, &ExtractJsonMiddleware{}))
	text := collectText(got)
	if text != `{"answer":"42"}` {
		t.Errorf("text = %q, want %q", text, `{"answer":"42"}`)
	}
}

func TestExtractJson_NoFencePassesThrough(t *testing.T) {
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: `{"raw":true}`}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{}},
	})
	got := drainAll(t, WrapModel(inner, &ExtractJsonMiddleware{}))
	text := collectText(got)
	if text != `{"raw":true}` {
		t.Errorf("text = %q, want %q", text, `{"raw":true}`)
	}
}

func TestExtractJson_CustomTransformBuffersAll(t *testing.T) {
	calls := 0
	mw := &ExtractJsonMiddleware{
		Transform: func(s string) string {
			calls++
			// Trim any wrapping parens.
			return strings.TrimSuffix(strings.TrimPrefix(s, "("), ")")
		},
	}
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "(hello "}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "world)"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{}},
	})
	got := drainAll(t, WrapModel(inner, mw))
	if calls != 1 {
		t.Errorf("transform called %d times, want 1", calls)
	}
	if text := collectText(got); text != "hello world" {
		t.Errorf("text = %q, want 'hello world'", text)
	}
}

func TestExtractJson_SplitFenceAcrossDeltas(t *testing.T) {
	// ` ```json\n` arriving split across two deltas — the middleware should
	// still detect the fence once the newline lands.
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "```"}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "json\n"}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: `{"k":"v"}`}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "\n```"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{}},
	})
	got := drainAll(t, WrapModel(inner, &ExtractJsonMiddleware{}))
	text := collectText(got)
	if text != `{"k":"v"}` {
		t.Errorf("text = %q, want %q", text, `{"k":"v"}`)
	}
}

func TestExtractJson_NonTextEventsPassThrough(t *testing.T) {
	tc := stream.ToolCallEvent{ToolCallID: "x", ToolName: "t"}
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "a"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventToolCall, Data: tc},
		{Type: stream.EventFinish, Data: stream.FinishEvent{}},
	})
	got := drainAll(t, WrapModel(inner, &ExtractJsonMiddleware{}))
	var sawTool bool
	for _, ev := range got {
		if ev.Type == stream.EventToolCall {
			sawTool = true
		}
	}
	if !sawTool {
		t.Error("ToolCall should pass through")
	}
}
