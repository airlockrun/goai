package middleware

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/stream"
)

// scripted builds a stub stream.Model whose Stream sends the given events
// in order and then closes the channel. Used for testing middlewares that
// wrap a streaming model.
func scripted(events []stream.Event) stream.Model {
	return scriptedModel{events: events}
}

type scriptedModel struct {
	events []stream.Event
}

func (scriptedModel) ID() string       { return "scripted" }
func (scriptedModel) Provider() string { return "scripted" }

func (m scriptedModel) Stream(_ context.Context, _ *stream.CallOptions) (<-chan stream.Event, error) {
	ch := make(chan stream.Event, len(m.events))
	for _, ev := range m.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func drainAll(t *testing.T, m stream.Model) []stream.Event {
	t.Helper()
	evs, err := m.Stream(context.Background(), &stream.CallOptions{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var out []stream.Event
	for ev := range evs {
		out = append(out, ev)
	}
	return out
}

func TestSimulateStreaming_TextRoundTrip(t *testing.T) {
	inner := scripted([]stream.Event{
		{Type: stream.EventStart, Data: stream.StartEvent{}},
		{Type: stream.EventStartStep, Data: stream.StartStepEvent{}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "Hello "}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "world"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinishStep, Data: stream.FinishStepEvent{FinishReason: stream.FinishReasonStop, Usage: stream.UsageFrom(3, 2)}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop, Usage: stream.UsageFrom(3, 2)}},
	})
	wrapped := WrapModel(inner, &SimulateStreamingMiddleware{})
	got := drainAll(t, wrapped)

	// Expected: start, start-step, text-start, single text-delta ("Hello world"),
	// text-end, finish-step, finish.
	if len(got) != 7 {
		t.Fatalf("got %d events, want 7: %+v", len(got), got)
	}
	types := []stream.EventType{
		stream.EventStart, stream.EventStartStep,
		stream.EventTextStart, stream.EventTextDelta, stream.EventTextEnd,
		stream.EventFinishStep, stream.EventFinish,
	}
	for i, want := range types {
		if got[i].Type != want {
			t.Errorf("event %d: type = %s, want %s", i, got[i].Type, want)
		}
	}
	delta := got[3].Data.(stream.TextDeltaEvent)
	if delta.Text != "Hello world" {
		t.Errorf("text-delta = %q, want %q", delta.Text, "Hello world")
	}
	finish := got[6].Data.(stream.FinishEvent)
	if finish.FinishReason != stream.FinishReasonStop {
		t.Errorf("FinishReason = %s", finish.FinishReason)
	}
	if finish.Usage.GrandTotal() != 5 {
		t.Errorf("Usage.TotalTokens = %d, want 5", finish.Usage.GrandTotal())
	}
}

func TestSimulateStreaming_ReasoningAndText(t *testing.T) {
	inner := scripted([]stream.Event{
		{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "r1"}},
		{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "r1", Text: "Thinking..."}},
		{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "r1"}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "Answer"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop}},
	})
	got := drainAll(t, WrapModel(inner, &SimulateStreamingMiddleware{}))

	// Verify reasoning block comes before text block.
	var rIdx, tIdx int = -1, -1
	for i, ev := range got {
		if ev.Type == stream.EventReasoningDelta {
			rIdx = i
		}
		if ev.Type == stream.EventTextDelta {
			tIdx = i
		}
	}
	if rIdx < 0 || tIdx < 0 || rIdx > tIdx {
		t.Errorf("reasoning should appear before text, got reasoningIdx=%d textIdx=%d", rIdx, tIdx)
	}
}

func TestSimulateStreaming_ToolCallsPassThrough(t *testing.T) {
	toolCall := stream.ToolCallEvent{ToolCallID: "c1", ToolName: "t", Input: json.RawMessage(`{"x":1}`)}
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "ok"}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		{Type: stream.EventToolCall, Data: toolCall},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonToolCalls}},
	})
	got := drainAll(t, WrapModel(inner, &SimulateStreamingMiddleware{}))

	found := false
	for _, ev := range got {
		if ev.Type == stream.EventToolCall {
			tc := ev.Data.(stream.ToolCallEvent)
			if tc.ToolCallID == "c1" && tc.ToolName == "t" {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected ToolCall to pass through unchanged")
	}
}

func TestSimulateStreaming_ErrorPropagates(t *testing.T) {
	boom := &stream.ErrorEvent{Error: context.DeadlineExceeded}
	inner := scripted([]stream.Event{
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventError, Data: *boom},
	})
	got := drainAll(t, WrapModel(inner, &SimulateStreamingMiddleware{}))

	var sawErr bool
	for _, ev := range got {
		if ev.Type == stream.EventError {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("expected EventError to be re-emitted")
	}
}
