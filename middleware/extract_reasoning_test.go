package middleware

import (
	"strings"
	"testing"

	"github.com/airlockrun/goai/stream"
)

// collectReasoning concatenates all ReasoningDeltaEvent.Text.
func collectReasoning(events []stream.Event) string {
	var sb strings.Builder
	for _, ev := range events {
		if d, ok := ev.Data.(stream.ReasoningDeltaEvent); ok {
			sb.WriteString(d.Text)
		}
	}
	return sb.String()
}

// streamText emits TextStart, a series of TextDelta events for each slice,
// then TextEnd and Finish.
func streamText(deltas ...string) stream.Model {
	evs := []stream.Event{{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}}
	for _, d := range deltas {
		evs = append(evs, stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: d}})
	}
	evs = append(evs,
		stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
		stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop}},
	)
	return scripted(evs)
}

func TestExtractReasoning_BasicTagPair(t *testing.T) {
	inner := streamText("<think>pondering</think>Answer is 42")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think"}))

	if collectReasoning(got) != "pondering" {
		t.Errorf("reasoning = %q, want %q", collectReasoning(got), "pondering")
	}
	if collectText(got) != "Answer is 42" {
		t.Errorf("text = %q, want %q", collectText(got), "Answer is 42")
	}
}

func TestExtractReasoning_SplitTagAcrossDeltas(t *testing.T) {
	// Tag splits across delta boundaries — the middleware should not leak
	// a partial `<thi` into text.
	inner := streamText("pre<thi", "nk>hidden</thi", "nk>post")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think"}))
	if r := collectReasoning(got); r != "hidden" {
		t.Errorf("reasoning = %q, want %q", r, "hidden")
	}
	// ai-sdk inserts the separator when text resumes after a reasoning
	// interlude — "pre" and "post" span a tag boundary so "\n" lands between.
	if txt := collectText(got); txt != "pre\npost" {
		t.Errorf("text = %q, want %q", txt, "pre\npost")
	}
}

func TestExtractReasoning_MultipleTagPairs(t *testing.T) {
	inner := streamText("<think>a</think>x<think>b</think>y")
	mw := &ExtractReasoningMiddleware{TagName: "think", Separator: "|"}
	got := drainAll(t, WrapModel(inner, mw))

	// Two reasoning blocks separated by switches — each emits its own
	// Start/End pair; the "|" separator kicks in when reasoning resumes.
	if r := collectReasoning(got); r != "a|b" {
		t.Errorf("reasoning = %q, want %q", r, "a|b")
	}
	if txt := collectText(got); txt != "x|y" {
		t.Errorf("text = %q, want %q", txt, "x|y")
	}

	// Should see two ReasoningStart events.
	var starts int
	for _, ev := range got {
		if ev.Type == stream.EventReasoningStart {
			starts++
		}
	}
	if starts != 2 {
		t.Errorf("ReasoningStart count = %d, want 2", starts)
	}
}

func TestExtractReasoning_StartWithReasoning(t *testing.T) {
	// No opening tag — stream begins inside reasoning per the flag.
	inner := streamText("musing</think>final")
	mw := &ExtractReasoningMiddleware{TagName: "think", StartWithReasoning: true}
	got := drainAll(t, WrapModel(inner, mw))

	if r := collectReasoning(got); r != "musing" {
		t.Errorf("reasoning = %q, want %q", r, "musing")
	}
	if txt := collectText(got); txt != "final" {
		t.Errorf("text = %q, want %q", txt, "final")
	}
}

func TestExtractReasoning_NoTagPassesThrough(t *testing.T) {
	inner := streamText("no reasoning here")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think"}))

	if collectReasoning(got) != "" {
		t.Error("expected no reasoning output")
	}
	if txt := collectText(got); txt != "no reasoning here" {
		t.Errorf("text = %q, want %q", txt, "no reasoning here")
	}
}

// Regression test for ai-sdk #12055: empty reasoning block like
// <think></think> emitted ReasoningEnd without a preceding ReasoningStart,
// causing downstream "reasoning-0 not found" errors.
func TestExtractReasoning_EmptyReasoningBlock(t *testing.T) {
	inner := streamText("<think></think>After")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think"}))

	if collectText(got) != "After" {
		t.Errorf("text = %q, want %q", collectText(got), "After")
	}

	var starts, ends int
	for _, ev := range got {
		switch ev.Type {
		case stream.EventReasoningStart:
			starts++
		case stream.EventReasoningEnd:
			ends++
		}
	}
	if starts != 1 || ends != 1 {
		t.Errorf("ReasoningStart/End = %d/%d, want 1/1 (balanced pair)", starts, ends)
	}
}

// Regression for the StartWithReasoning=true variant: </think> immediately
// at the start of the stream means the opening tag is implicit, so the
// first tag we see is the close. If no delta was published before it, we
// still need a balanced ReasoningStart/End pair.
func TestExtractReasoning_ImmediateCloseWithStartFlag(t *testing.T) {
	inner := streamText("</think>After")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think", StartWithReasoning: true}))

	if collectText(got) != "After" {
		t.Errorf("text = %q, want %q", collectText(got), "After")
	}

	var starts, ends int
	for _, ev := range got {
		switch ev.Type {
		case stream.EventReasoningStart:
			starts++
		case stream.EventReasoningEnd:
			ends++
		}
	}
	if starts != 1 || ends != 1 {
		t.Errorf("ReasoningStart/End = %d/%d, want 1/1", starts, ends)
	}
}

func TestExtractReasoning_ReasoningOnlyClosesCleanly(t *testing.T) {
	// Reasoning block with no trailing text: make sure ReasoningEnd fires
	// and TextStart isn't emitted.
	inner := streamText("<think>thinking</think>")
	got := drainAll(t, WrapModel(inner, &ExtractReasoningMiddleware{TagName: "think"}))

	if collectReasoning(got) != "thinking" {
		t.Errorf("reasoning = %q", collectReasoning(got))
	}
	if txt := collectText(got); txt != "" {
		t.Errorf("text = %q, want empty", txt)
	}

	var starts, ends int
	for _, ev := range got {
		switch ev.Type {
		case stream.EventReasoningStart:
			starts++
		case stream.EventReasoningEnd:
			ends++
		}
	}
	if starts != 1 || ends != 1 {
		t.Errorf("ReasoningStart/End = %d/%d, want 1/1", starts, ends)
	}
}
