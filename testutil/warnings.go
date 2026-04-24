package testutil

import (
	"testing"

	"github.com/airlockrun/goai/stream"
)

// FindFirstStartEvent returns the first StartEvent in events. Fails the
// test if no StartEvent is present.
func FindFirstStartEvent(t *testing.T, events []stream.Event) *stream.StartEvent {
	t.Helper()
	for _, ev := range events {
		if ev.Type == stream.EventStart {
			if s, ok := ev.Data.(stream.StartEvent); ok {
				return &s
			}
		}
	}
	t.Fatalf("no StartEvent in %d events", len(events))
	return nil
}

// AssertWarning fails the test if warnings does not contain want.
// Matches on Type + Feature + Message; Details is ignored.
func AssertWarning(t *testing.T, warnings []stream.Warning, want stream.Warning) {
	t.Helper()
	for _, w := range warnings {
		if w.Type == want.Type && w.Feature == want.Feature && w.Message == want.Message {
			return
		}
	}
	t.Errorf("missing warning %+v in %+v", want, warnings)
}

// AssertResultWarning is an alias for AssertWarning, named to match
// non-language-model result types (ImageResult.Warnings, etc.).
func AssertResultWarning(t *testing.T, warnings []stream.Warning, want stream.Warning) {
	t.Helper()
	AssertWarning(t, warnings, want)
}

// CollectStream drains a stream channel into a slice.
func CollectStream(t *testing.T, ch <-chan stream.Event, err error) []stream.Event {
	t.Helper()
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var out []stream.Event
	for ev := range ch {
		out = append(out, ev)
	}
	return out
}
