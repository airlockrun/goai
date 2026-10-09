package testutil

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func newTestMock(t testing.TB, config MockConfig) *MockModel {
	t.Helper()
	m, err := NewMockModel(config)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func collectText(events <-chan stream.Event) string {
	var b strings.Builder
	for event := range events {
		if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
			b.WriteString(delta.Text)
		}
	}
	return b.String()
}

func mockEvents(t *testing.T, m stream.Model, options *stream.CallOptions) []stream.Event {
	t.Helper()
	ch, err := m.Stream(t.Context(), options)
	if err != nil {
		t.Fatal(err)
	}
	var events []stream.Event
	for event := range ch {
		events = append(events, event)
	}
	return events
}

func TestMockMatching(t *testing.T) {
	exact, empty := "hello world", ""
	m := newTestMock(t, MockConfig{ID: "fixture", Default: &MockResponse{Text: "fallback"}, Rules: []MockRule{
		{LastUserText: &exact, Match: func(o *stream.CallOptions) bool { return o.ToolChoice == "none" }, Response: MockResponse{Text: "both"}},
		{LastUserText: &exact, Response: MockResponse{Text: "first exact"}},
		{Match: func(o *stream.CallOptions) bool { return len(o.Messages) > 0 }, Response: MockResponse{Text: "predicate"}},
		{LastUserText: &empty, Response: MockResponse{Text: "empty"}},
	}})
	for _, tt := range []struct {
		name    string
		options stream.CallOptions
		want    string
	}{
		{"fallback", stream.CallOptions{}, "fallback"},
		{"exact before predicate", stream.CallOptions{Messages: []message.Message{message.NewUserMessage(exact)}}, "first exact"},
		{"both", stream.CallOptions{Messages: []message.Message{message.NewUserMessage(exact)}, ToolChoice: "none"}, "both"},
		{"last user multipart", stream.CallOptions{Messages: []message.Message{message.NewUserMessage("old"), message.NewUserMessageWithParts(message.TextPart{Text: "hello "}, message.TextPart{Text: "world"}), message.NewAssistantMessage("other")}}, "first exact"},
		{"predicate", stream.CallOptions{Messages: []message.Message{message.NewUserMessage("different")}}, "predicate"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			events := mockEvents(t, m, &tt.options)
			if got := events[2].Data.(stream.TextDeltaEvent).Text; got != tt.want {
				t.Fatalf("text = %q, want %q", got, tt.want)
			}
		})
	}
	strict := newTestMock(t, MockConfig{ID: "strict", Rules: []MockRule{{LastUserText: &empty}}})
	if ch, err := strict.Stream(t.Context(), &stream.CallOptions{}); ch != nil || !errors.Is(err, ErrMockNoMatch) {
		t.Fatalf("unmatched = %v, %v", ch, err)
	}
	if len(strict.Requests()) != 1 {
		t.Fatal("unmatched request not captured")
	}
	mockEvents(t, strict, &stream.CallOptions{Messages: []message.Message{message.NewUserMessage("")}})
}

func TestMockValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		config MockConfig
	}{
		{"missing ID", MockConfig{}},
		{"missing matcher", MockConfig{ID: "m", Rules: []MockRule{{}}}},
		{"invalid response", MockConfig{ID: "m", Default: &MockResponse{ToolCalls: []stream.ToolCall{{Input: json.RawMessage("invalid")}}}}},
		{"nil event payload", MockConfig{ID: "m", Default: &MockResponse{Events: []stream.Event{{Type: stream.EventTextDelta, Data: (*stream.TextDeltaEvent)(nil)}}}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewMockModel(tt.config); err == nil {
				t.Fatal("expected error")
			}
		})
	}
	m := newTestMock(t, MockConfig{ID: "m", Default: &MockResponse{}})
	if _, err := m.Stream(t.Context(), nil); err == nil {
		t.Fatal("nil input accepted")
	}
	if _, err := m.Stream(t.Context(), &stream.CallOptions{ProviderOptions: map[string]any{"bad": make(chan int)}}); err == nil {
		t.Fatal("non-JSON input accepted")
	}
}

func TestMockTextEvents(t *testing.T) {
	m := newTestMock(t, MockConfig{ID: "fixture", Default: &MockResponse{Text: "answer", Usage: stream.UsageFrom(3, 2)}})
	if m.ID() != "fixture" || m.Provider() != "mock" {
		t.Fatal("incorrect identity")
	}
	events := mockEvents(t, m, &stream.CallOptions{})
	var types []stream.EventType
	for _, event := range events {
		types = append(types, event.Type)
	}
	if !reflect.DeepEqual(types, []stream.EventType{stream.EventStart, stream.EventTextStart, stream.EventTextDelta, stream.EventTextEnd, stream.EventFinish}) {
		t.Fatalf("events = %v", types)
	}
	finish := events[4].Data.(stream.FinishEvent)
	if finish.FinishReason != stream.FinishReasonStop || finish.Usage.InputTotal() != 3 || finish.Usage.OutputTotal() != 2 {
		t.Fatalf("finish = %+v", finish)
	}
	*finish.Usage.InputTokens.Total = 99
	if mockEvents(t, m, &stream.CallOptions{})[4].Data.(stream.FinishEvent).Usage.InputTotal() != 3 {
		t.Fatal("event mutated model")
	}
}

func TestMockCancellation(t *testing.T) {
	for _, abort := range []bool{false, true} {
		t.Run(map[bool]string{false: "context", true: "abort signal"}[abort], func(t *testing.T) {
			m := newTestMock(t, MockConfig{ID: "m", Default: &MockResponse{Text: "answer"}})
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			options := &stream.CallOptions{}
			callCtx := ctx
			if abort {
				callCtx = t.Context()
				options.AbortSignal = ctx
			}
			ch, err := m.Stream(callCtx, options)
			if err != nil {
				t.Fatal(err)
			}
			<-ch // Leave the producer blocked on the next event.
			cancel()
			// A pending send can win its select against cancellation. Drain any
			// in-flight event while requiring the stream to close promptly.
			deadline := time.After(time.Second)
			for closed := false; !closed; {
				select {
				case _, ok := <-ch:
					closed = !ok
				case <-deadline:
					t.Fatal("stream failed to close")
				}
			}
			if _, err := m.Stream(callCtx, options); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled setup = %v", err)
			}
			if len(m.Requests()) != 1 {
				t.Fatal("pre-canceled request captured")
			}
		})
	}
}

func TestMockRequestCopies(t *testing.T) {
	text := "original"
	config := MockConfig{ID: "m", Default: &MockResponse{Text: "default"}, Rules: make([]MockRule, 1)}
	// Decoded message parts have GoAI's canonical value representation.
	config.Rules[0].Match = func(o *stream.CallOptions) bool {
		o.Messages[0].Content.Parts[0] = message.TextPart{Text: "mutated"}
		o.Headers["key"] = "mutated"
		return false
	}
	config.Rules = append(config.Rules, MockRule{LastUserText: &text, Response: MockResponse{Text: "matched"}})
	m := newTestMock(t, config)
	text = "changed config"
	config.Rules[1].Response.Text = "changed response"
	options := stream.CallOptions{Messages: []message.Message{message.NewUserMessageWithParts(message.TextPart{Text: "original"}, message.FilePart{Data: message.FileDataReference{Reference: map[string]any{"id": "file"}}, MimeType: "application/octet-stream"})}, Headers: map[string]string{"key": "original"}, ProviderOptions: map[string]any{"nested": map[string]any{"value": "original"}}, Temperature: new(float64), Tools: []tool.Tool{{Name: "test", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	if mockEvents(t, m, &options)[2].Data.(stream.TextDeltaEvent).Text != "matched" {
		t.Fatal("predicate/config mutation affected match")
	}
	options.Headers["key"] = "caller changed"
	options.Tools[0].InputSchema[0] = ' '
	history := m.Requests()
	if history[0].Headers["key"] != "original" {
		t.Fatal("caller/predicate changed history")
	}
	history[0].Headers["key"] = "inspection changed"
	history[0].Messages[0].Content.Parts[1].(message.FilePart).Data.(message.FileDataReference).Reference["id"] = "changed"
	history[0].ProviderOptions["nested"].(map[string]any)["value"] = "changed"
	*history[0].Temperature = 99
	if got := m.Requests()[0]; got.Headers["key"] != "original" || *got.Temperature != 0 || got.Messages[0].Content.Parts[1].(message.FilePart).Data.(message.FileDataReference).Reference["id"] != "file" || got.ProviderOptions["nested"].(map[string]any)["value"] != "original" {
		t.Fatal("inspection changed history")
	}
	m.ResetRequests()
	if len(m.Requests()) != 0 {
		t.Fatal("reset failed")
	}
}

func TestMockConcurrentHistory(t *testing.T) {
	m := newTestMock(t, MockConfig{ID: "m", Default: &MockResponse{Text: "answer"}})
	var wg sync.WaitGroup
	for range 32 {
		wg.Go(func() {
			mockEvents(t, m, &stream.CallOptions{Messages: []message.Message{message.NewUserMessage("request")}})
			m.Requests()
		})
	}
	wg.Wait()
	if len(m.Requests()) != 32 {
		t.Fatalf("captured %d requests", len(m.Requests()))
	}
	for range 16 {
		wg.Go(func() { m.ResetRequests(); mockEvents(t, m, &stream.CallOptions{}); m.Requests() })
	}
	wg.Wait()
}

func TestMockGoAIInteroperability(t *testing.T) {
	m := newTestMock(t, MockConfig{ID: "agent", Default: &MockResponse{Text: "done", Usage: stream.UsageFrom(2, 1)}, Rules: []MockRule{{
		Match:    func(o *stream.CallOptions) bool { return o.Messages[len(o.Messages)-1].Role == message.RoleUser },
		Response: MockResponse{ToolCalls: []stream.ToolCall{{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)}}, Usage: stream.UsageFrom(3, 1)},
	}}})
	calls := 0
	lookup := tool.New("lookup").Execute(func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
		calls++
		if opts.ToolCallID != "call-1" {
			t.Errorf("call ID = %q", opts.ToolCallID)
		}
		return tool.Result{Output: "found"}, nil
	}).Build()
	result, err := goai.GenerateText(t.Context(), stream.Input{Model: m, Messages: []message.Message{message.NewUserMessage("find")}, Tools: tool.Set{"lookup": lookup}, MaxSteps: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" || calls != 1 || len(m.Requests()) != 2 || result.Usage.InputTotal() != 5 {
		t.Fatalf("result = %+v, tool calls = %d", result, calls)
	}
	streamed, err := goai.StreamText(t.Context(), stream.Input{Model: m, Messages: []message.Message{message.NewAssistantMessage("continue")}})
	if err != nil {
		t.Fatal(err)
	}
	for range streamed.FullStream {
	}
	if text, err := streamed.Text(); err != nil || text != "done" {
		t.Fatalf("streamed text = %q, %v", text, err)
	}
}

func TestMockSequenceAndEvents(t *testing.T) {
	errScript := errors.New("scripted failure")
	text := "rule"
	script := []stream.Event{
		{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "bad", ToolName: "tool", Input: json.RawMessage(`{`), ProviderMetadata: map[string]any{"key": "original"}}},
		{Type: stream.EventError, Data: stream.ErrorEvent{Error: errScript}},
	}
	m := newTestMock(t, MockConfig{ID: "sequence", Default: &MockResponse{Text: "unused"},
		Rules:     []MockRule{{LastUserText: &text, Response: MockResponse{Text: "matched"}}},
		Responses: []MockResponse{{Text: "first"}, {Events: script}},
	})
	script[0].Data.(stream.ToolCallEvent).Input[0] = '!'
	if got := mockEvents(t, m, &stream.CallOptions{Messages: []message.Message{message.NewUserMessage(text)}})[2].Data.(stream.TextDeltaEvent).Text; got != "matched" {
		t.Fatalf("rule response = %q", got)
	}
	if got := mockEvents(t, m, &stream.CallOptions{})[2].Data.(stream.TextDeltaEvent).Text; got != "first" {
		t.Fatalf("first fallback = %q", got)
	}
	for range 2 {
		events := mockEvents(t, m, &stream.CallOptions{})
		if len(events) != 2 || string(events[0].Data.(stream.ToolCallEvent).Input) != "{" || !errors.Is(events[1].Data.(stream.ErrorEvent).Error, errScript) {
			t.Fatalf("script = %+v", events)
		}
		events[0].Data.(stream.ToolCallEvent).Input[0] = '!'
		events[0].Data.(stream.ToolCallEvent).ProviderMetadata["key"] = "changed"
	}
	if err := m.Configure(MockConfig{ID: "sequence", Default: &MockResponse{Events: []stream.Event{}}}); err != nil {
		t.Fatal(err)
	}
	if len(mockEvents(t, m, &stream.CallOptions{})) != 0 {
		t.Fatal("explicit empty event response generated events")
	}
}

func TestMockConfigure(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	m := newTestMock(t, MockConfig{ID: "configured", Rules: []MockRule{{
		Match:    func(*stream.CallOptions) bool { close(entered); <-release; return true },
		Response: MockResponse{Text: "in flight"},
	}}})
	old := make(chan string, 1)
	go func() {
		ch, err := m.Stream(t.Context(), &stream.CallOptions{})
		if err != nil {
			old <- err.Error()
			return
		}
		old <- collectText(ch)
	}()
	<-entered
	config := MockConfig{ID: "configured", Responses: []MockResponse{{Text: "one"}, {Text: "two"}}}
	if err := m.Configure(config); err != nil {
		t.Fatal(err)
	}
	close(release)
	if got := <-old; got != "in flight" {
		t.Fatalf("in-flight response = %q", got)
	}
	config.Responses[0].Text = "caller mutation"
	if got := mockEvents(t, m, &stream.CallOptions{})[2].Data.(stream.TextDeltaEvent).Text; got != "one" {
		t.Fatalf("configured response = %q", got)
	}
	for _, invalid := range []MockConfig{
		{ID: "other"},
		{ID: "configured", Rules: []MockRule{{}}},
		{ID: "configured", Responses: []MockResponse{{Usage: stream.Usage{Raw: map[string]any{"bad": make(chan int)}}}}},
	} {
		if err := m.Configure(invalid); err == nil {
			t.Fatal("invalid configuration accepted")
		}
	}
	if got := mockEvents(t, m, &stream.CallOptions{})[2].Data.(stream.TextDeltaEvent).Text; got != "two" {
		t.Fatalf("failed Configure changed response = %q", got)
	}
	if m.ID() != "configured" || len(m.Requests()) != 3 {
		t.Fatal("Configure changed ID/history")
	}
	if err := m.Configure(MockConfig{ID: "configured", Responses: []MockResponse{{Text: "restart"}, {Text: "last"}}}); err != nil {
		t.Fatal(err)
	}
	m.ResetRequests()
	if got := mockEvents(t, m, &stream.CallOptions{})[2].Data.(stream.TextDeltaEvent).Text; got != "restart" {
		t.Fatalf("sequence not restarted = %q", got)
	}
}

func TestMockConcurrentConfigure(t *testing.T) {
	m := newTestMock(t, MockConfig{ID: "concurrent", Default: &MockResponse{Text: "a"}})
	var wg sync.WaitGroup
	for range 24 {
		wg.Go(func() {
			for range 8 {
				if err := m.Configure(MockConfig{ID: "concurrent", Responses: []MockResponse{{Text: "a"}, {Text: "b"}}}); err != nil {
					t.Error(err)
					return
				}
				ch, err := m.Stream(t.Context(), &stream.CallOptions{})
				if err != nil {
					t.Error(err)
					return
				}
				if text := collectText(ch); text != "a" && text != "b" {
					t.Errorf("mixed configuration: %q", text)
				}
				m.Requests()
			}
		})
	}
	wg.Wait()
	if len(m.Requests()) != 24*8 {
		t.Fatalf("captured %d requests", len(m.Requests()))
	}
}

func TestMockConcurrentStreamHookConfigure(t *testing.T) {
	hook := func(ctx context.Context, input *stream.CallOptions) (<-chan stream.Event, error) {
		input.Headers["key"] = "hook mutation"
		return mockEmit(ctx, nil, MockTextResponse("hook", stream.Usage{})), nil
	}
	m := newTestMock(t, MockConfig{ID: "concurrent hook", Stream: hook})
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for i := range 8 {
				config := MockConfig{ID: m.ID(), Default: &MockResponse{Text: "script"}}
				if i%2 == 0 {
					config.Stream = hook
				}
				if err := m.Configure(config); err != nil {
					t.Error(err)
					return
				}
				input := &stream.CallOptions{Headers: map[string]string{"key": "original"}}
				ch, err := m.Stream(t.Context(), input)
				if err != nil {
					t.Error(err)
					return
				}
				if text := collectText(ch); text != "script" && text != "hook" {
					t.Errorf("text = %q", text)
				}
				if input.Headers["key"] != "original" {
					t.Error("hook mutated input")
				}
				for _, request := range m.Requests() {
					if request.Headers["key"] != "original" {
						t.Error("hook mutated history")
					}
				}
			}
		})
	}
	wg.Wait()
	if len(m.Requests()) != 16*8 {
		t.Fatalf("captured %d requests", len(m.Requests()))
	}
}

func TestMockEventUnionCopies(t *testing.T) {
	script := []stream.Event{
		{Type: stream.EventReasoningDelta, Data: &stream.ReasoningDeltaEvent{ID: "reason", Text: "thinking"}},
		{Type: stream.EventToolResult, Data: stream.ToolResultEvent{ToolCallID: "call", ToolName: "lookup", Output: message.JSONOutput{Value: map[string]any{"found": true}}, Metadata: map[string]any{"label": "original"}}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop, Usage: MockUsage(2, 1)}},
	}
	m := newTestMock(t, MockConfig{ID: "union", Default: &MockResponse{Events: script}})
	script[0].Data.(*stream.ReasoningDeltaEvent).Text = "config changed"
	for range 2 {
		events := mockEvents(t, m, &stream.CallOptions{})
		if events[0].Data.(*stream.ReasoningDeltaEvent).Text != "thinking" {
			t.Fatal("pointer event was not copied")
		}
		result := events[1].Data.(stream.ToolResultEvent)
		if result.Output.(message.JSONOutput).Value.(map[string]any)["found"] != true || result.Metadata["label"] != "original" {
			t.Fatal("tool-result union was not copied")
		}
		result.Output.(message.JSONOutput).Value.(map[string]any)["found"] = false
		result.Metadata["label"] = "changed"
		*events[2].Data.(stream.FinishEvent).Usage.InputTokens.Total = 99
		if events[2].Data.(stream.FinishEvent).FinishReason != stream.FinishReasonStop {
			t.Fatal("finish discriminator changed")
		}
	}
}

func TestMockStreamHookIsolationAndPriority(t *testing.T) {
	m := newTestMock(t, MockConfig{ID: "hook",
		Default: &MockResponse{Text: "default"}, Responses: []MockResponse{{Text: "sequence"}},
		Rules: []MockRule{{Match: func(*stream.CallOptions) bool { t.Error("rule invoked with hook configured"); return true }}},
		Stream: func(ctx context.Context, input *stream.CallOptions) (<-chan stream.Event, error) {
			input.Headers["key"] = "hook changed"
			input.Messages[0].Content.Text = "hook changed"
			return mockEmit(ctx, nil, MockTextResponse("hook", MockUsage(1, 1))), nil
		},
	})
	input := stream.CallOptions{Headers: map[string]string{"key": "original"}, Messages: []message.Message{message.NewUserMessage("original")}}
	ch, err := m.Stream(t.Context(), &input)
	if err != nil {
		t.Fatal(err)
	}
	if got := collectText(ch); got != "hook" {
		t.Fatalf("text = %q", got)
	}
	if input.Headers["key"] != "original" || input.Messages[0].Content.Text != "original" {
		t.Fatal("hook mutated caller")
	}
	history := m.Requests()
	if len(history) != 1 || history[0].Headers["key"] != "original" || history[0].Messages[0].Content.Text != "original" {
		t.Fatal("hook mutated history")
	}
	history[0].Headers["key"] = "inspection changed"
	if m.Requests()[0].Headers["key"] != "original" {
		t.Fatal("history is not isolated")
	}
	m.ResetRequests()
	if len(m.Requests()) != 0 {
		t.Fatal("reset failed")
	}
}

func TestMockStreamHookErrors(t *testing.T) {
	failure := errors.New("hook setup failed")
	for _, tt := range []struct {
		name    string
		failure error
		want    string
	}{
		{"setup error", failure, "hook setup failed"},
		{"nil channel", nil, "nil event stream"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var hookCtx context.Context
			m := newTestMock(t, MockConfig{ID: "errors", Stream: func(ctx context.Context, _ *stream.CallOptions) (<-chan stream.Event, error) {
				hookCtx = ctx
				return nil, tt.failure
			}})
			ch, err := m.Stream(t.Context(), &stream.CallOptions{})
			if ch != nil || err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Stream = %v, %v", ch, err)
			}
			if tt.failure != nil && !errors.Is(err, tt.failure) {
				t.Fatal("lost setup error identity")
			}
			if hookCtx.Err() == nil || len(m.Requests()) != 1 {
				t.Fatal("failed setup did not clean up or capture request")
			}
		})
	}
}

func TestMockStreamHookCancellation(t *testing.T) {
	for _, source := range []string{"context", "abort signal"} {
		for _, phase := range []string{"setup", "idle stream", "blocked forwarding"} {
			t.Run(source+"/"+phase, func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				callCtx, options := ctx, &stream.CallOptions{}
				if source == "abort signal" {
					callCtx = t.Context()
					options.AbortSignal = ctx
				}
				var hookCtx context.Context
				m := newTestMock(t, MockConfig{ID: "cancel", Stream: func(ctx context.Context, _ *stream.CallOptions) (<-chan stream.Event, error) {
					hookCtx = ctx
					if phase == "setup" {
						cancel()
						<-ctx.Done()
						return nil, ctx.Err()
					}
					ch := make(chan stream.Event, 1)
					if phase == "blocked forwarding" {
						ch <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "pending"}}
					}
					return ch, nil
				}})
				ch, err := m.Stream(callCtx, options)
				if phase == "setup" {
					if ch != nil || !errors.Is(err, context.Canceled) {
						t.Fatalf("Stream = %v, %v", ch, err)
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
					cancel()
					deadline := time.After(time.Second)
					for closed := false; !closed; {
						select {
						case _, ok := <-ch:
							closed = !ok
						case <-deadline:
							t.Fatal("canceled stream did not close")
						}
					}
				}
				if hookCtx.Err() == nil {
					t.Fatal("hook context not canceled")
				}
				if _, err := m.Stream(callCtx, options); !errors.Is(err, context.Canceled) {
					t.Fatalf("pre-canceled setup = %v", err)
				}
				if len(m.Requests()) != 1 {
					t.Fatal("pre-canceled request captured")
				}
			})
		}
	}
}

func TestMockStreamHookDelayedConfigure(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var hookCtx context.Context
	m := newTestMock(t, MockConfig{ID: "delayed", Stream: func(ctx context.Context, _ *stream.CallOptions) (<-chan stream.Event, error) {
		hookCtx = ctx
		ch := make(chan stream.Event)
		go func() {
			defer close(ch)
			close(entered)
			select {
			case <-release:
			case <-ctx.Done():
				return
			}
			select {
			case ch <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "captured hook"}}:
			case <-ctx.Done():
			}
		}()
		return ch, nil
	}})
	ch, err := m.Stream(t.Context(), &stream.CallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	<-entered
	select {
	case event := <-ch:
		t.Fatalf("event before release: %+v", event)
	default:
	}
	if err := m.Configure(MockConfig{ID: "delayed", Default: &MockResponse{Text: "configured"}}); err != nil {
		t.Fatal(err)
	}
	close(release)
	if got := collectText(ch); got != "captured hook" {
		t.Fatalf("in-flight text = %q", got)
	}
	if hookCtx.Err() == nil {
		t.Fatal("completed hook context not canceled")
	}
	if got := mockEvents(t, m, &stream.CallOptions{})[2].Data.(stream.TextDeltaEvent).Text; got != "configured" {
		t.Fatalf("new text = %q", got)
	}
	if len(m.Requests()) != 2 {
		t.Fatal("Configure changed history")
	}
}
