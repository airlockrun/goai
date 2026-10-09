package testutil

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// ErrMockNoMatch means no rule matched and no fallback response was configured.
var ErrMockNoMatch = errors.New("mock model: no matching response")

// MockResponse emits text, followed by optional tool calls and a finish event.
// Usage is supplied explicitly; zero Usage leaves token counts unreported.
type MockResponse struct {
	Text      string
	ToolCalls []stream.ToolCall
	Usage     stream.Usage
	// Events, when non-nil, replaces generated text/tool/finish events entirely.
	// An empty non-nil slice produces an empty stream.
	Events []stream.Event
}

// MockRule matches the last user message's text and/or a predicate. When both
// are set, both must match. Predicates receive an isolated request snapshot.
type MockRule struct {
	LastUserText *string
	Match        func(*stream.CallOptions) bool
	Response     MockResponse
}

// MockConfig configures a deterministic in-process model. Rules are checked in
// order; Responses and Default provide fallback when none matches.
type MockConfig struct {
	ID      string
	Default *MockResponse
	Rules   []MockRule
	// Responses is the fallback sequence when no rule matches. It takes
	// precedence over Default and repeats its last response when exhausted.
	Responses []MockResponse
	// Stream takes precedence over Rules, Responses, and Default. It receives
	// an isolated request and a context canceled by either the call context or
	// AbortSignal. Hooks own their channel and must honor context cancellation.
	Stream func(context.Context, *stream.CallOptions) (<-chan stream.Event, error)
}

// MockModel implements stream.Model without credentials or network access.
// Configure, Requests, and ResetRequests are safe to use concurrently with Stream.
type MockModel struct {
	id       string
	mu       sync.Mutex
	behavior *mockBehavior
	requests []stream.CallOptions
}

type mockBehavior struct {
	stream    func(context.Context, *stream.CallOptions) (<-chan stream.Event, error)
	def       *MockResponse
	rules     []MockRule
	responses []MockResponse
	next      int // Protected by MockModel.mu, including for captured configurations.
}

var _ stream.Model = (*MockModel)(nil)

// NewMockModel copies configuration and rejects rules without a matcher.
// ID must be explicit. Request/response data must be JSON-serializable.
func NewMockModel(config MockConfig) (*MockModel, error) {
	if config.ID == "" {
		return nil, errors.New("mock model: ID is required")
	}
	m := &MockModel{id: config.ID}
	if err := m.Configure(config); err != nil {
		return nil, err
	}
	return m, nil
}

// Configure atomically replaces response behavior and restarts its sequence.
// ID must equal the model's immutable ID. History is preserved. In-flight
// streams retain their captured configuration; invalid changes have no effect.
func (m *MockModel) Configure(config MockConfig) error {
	if config.ID != m.id {
		return errors.New("mock model: Configure cannot change ID")
	}
	behavior := &mockBehavior{stream: config.Stream}
	if config.Default != nil {
		response, err := mockCopyResponse(*config.Default)
		if err != nil {
			return fmt.Errorf("mock default: %w", err)
		}
		behavior.def = &response
	}
	for i, rule := range config.Rules {
		if rule.LastUserText == nil && rule.Match == nil {
			return fmt.Errorf("mock rule %d: matcher is required", i)
		}
		if rule.LastUserText != nil {
			text := *rule.LastUserText
			rule.LastUserText = &text
		}
		response, err := mockCopyResponse(rule.Response)
		if err != nil {
			return fmt.Errorf("mock rule %d: %w", i, err)
		}
		rule.Response = response
		behavior.rules = append(behavior.rules, rule)
	}
	for i, response := range config.Responses {
		copy, err := mockCopyResponse(response)
		if err != nil {
			return fmt.Errorf("mock response %d: %w", i, err)
		}
		behavior.responses = append(behavior.responses, copy)
	}
	m.mu.Lock()
	m.behavior = behavior
	m.mu.Unlock()
	return nil
}

func (m *MockModel) ID() string       { return m.id }
func (m *MockModel) Provider() string { return "mock" }

// Stream captures valid requests, including unmatched ones. Canceled requests
// are rejected before capture. Cancellation during emission closes the stream;
// consumers should check their context, as with other stream.Model providers.
func (m *MockModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options == nil {
		return nil, errors.New("mock model: CallOptions is required")
	}
	if options.AbortSignal != nil && options.AbortSignal.Err() != nil {
		return nil, options.AbortSignal.Err()
	}
	request, err := mockCopyRequest(*options)
	if err != nil {
		return nil, fmt.Errorf("mock request: %w", err)
	}
	m.mu.Lock()
	m.requests = append(m.requests, request)
	behavior := m.behavior
	m.mu.Unlock()

	if behavior.stream != nil {
		candidate, err := mockCopyRequest(request)
		if err != nil {
			return nil, err
		}
		hookCtx, cancel := context.WithCancel(ctx)
		stop := func() bool { return false }
		if request.AbortSignal != nil {
			stop = context.AfterFunc(request.AbortSignal, cancel)
		}
		cleanup := func() { stop(); cancel() }
		ch, err := behavior.stream(hookCtx, &candidate)
		if err != nil {
			cleanup()
			return nil, err
		}
		if err := hookCtx.Err(); err != nil {
			cleanup()
			return nil, err
		}
		if ch == nil {
			cleanup()
			return nil, errors.New("mock model: Stream hook returned nil event stream")
		}
		out := make(chan stream.Event)
		go func() {
			defer close(out)
			defer cleanup()
			for {
				select {
				case <-hookCtx.Done():
					return
				case event, ok := <-ch:
					if !ok {
						return
					}
					select {
					case <-hookCtx.Done():
						return
					case out <- event:
					}
				}
			}
		}()
		return out, nil
	}
	var response *MockResponse
	text, hasUser := mockLastUserText(request.Messages)
	for _, rule := range behavior.rules {
		if rule.LastUserText != nil && (!hasUser || text != *rule.LastUserText) {
			continue
		}
		if rule.Match != nil {
			candidate, err := mockCopyRequest(request)
			if err != nil {
				return nil, err
			}
			if !rule.Match(&candidate) {
				continue
			}
		}
		response = &rule.Response
		break
	}
	if response == nil {
		m.mu.Lock()
		response = mockNextResponse(behavior.responses, &behavior.next)
		m.mu.Unlock()
		if response == nil {
			response = behavior.def
		}
	}
	if response == nil {
		return nil, ErrMockNoMatch
	}
	// Each stream owns its mutable event payloads.
	owned, err := mockCopyResponse(*response)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if request.AbortSignal != nil && request.AbortSignal.Err() != nil {
		return nil, request.AbortSignal.Err()
	}
	events := mockResponseEvents(owned)
	return mockEmit(ctx, request.AbortSignal, events), nil
}

func mockNextResponse(responses []MockResponse, next *int) *MockResponse {
	if len(responses) == 0 {
		return nil
	}
	index := min(*next, len(responses)-1)
	if *next < len(responses)-1 {
		*next++
	}
	return &responses[index]
}

func mockResponseEvents(owned MockResponse) []stream.Event {
	if owned.Events != nil {
		return owned.Events
	}
	reason := stream.FinishReasonStop
	if len(owned.ToolCalls) > 0 {
		reason = stream.FinishReasonToolCalls
	}
	events := []stream.Event{
		{Type: stream.EventStart, Data: stream.StartEvent{}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: owned.Text}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
	}
	for _, call := range owned.ToolCalls {
		events = append(events, stream.Event{Type: stream.EventToolCall, Data: stream.ToolCallEvent{
			ToolCallID: call.ID, ToolName: call.Name, Input: call.Input,
			ProviderExecuted: call.ProviderExecuted, ProviderMetadata: call.ProviderMetadata,
		}})
	}
	events = append(events, stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: reason, Usage: owned.Usage}})
	return events
}

func mockEmit(ctx, signal context.Context, events []stream.Event) <-chan stream.Event {
	ch := make(chan stream.Event)
	var abort <-chan struct{}
	if signal != nil {
		abort = signal.Done()
	}
	go func() {
		defer close(ch)
		for _, event := range events {
			if ctx.Err() != nil || (signal != nil && signal.Err() != nil) {
				return
			}
			select {
			case <-ctx.Done():
				return
			case <-abort:
				return
			case ch <- event:
			}
		}
	}()
	return ch
}

func mockCopyResponse(response MockResponse) (MockResponse, error) {
	events := response.Events
	response.Events = nil
	var copy MockResponse
	if err := mockJSONCopy(response, &copy); err != nil {
		return copy, err
	}
	if events != nil {
		copy.Events = make([]stream.Event, len(events))
	}
	for i, event := range events {
		copy.Events[i].Type = event.Type
		if event.Data == nil {
			continue
		}
		// Errors are opaque immutable values, not JSON objects to reconstruct.
		switch data := event.Data.(type) {
		case stream.ErrorEvent:
			copy.Events[i].Data = data
			continue
		case *stream.ErrorEvent:
			if data == nil {
				return copy, errors.New("mock event: nil error payload")
			}
			owned := *data
			copy.Events[i].Data = &owned
			continue
		}
		// EventData is a GoAI-owned union. Preserve its concrete discriminator
		// while copying nested JSON data, including tool-result output unions.
		typ := reflect.TypeOf(event.Data)
		pointer := typ.Kind() == reflect.Pointer
		if pointer {
			if reflect.ValueOf(event.Data).IsNil() {
				return copy, fmt.Errorf("mock event %d: nil payload", i)
			}
			typ = typ.Elem()
		}
		data := reflect.New(typ)
		payload := event.Data
		var rawInput []byte
		// Scripted calls can intentionally contain malformed arguments to test
		// consumers' validation/repair paths. RawMessage must not validate them.
		if typ == reflect.TypeOf(stream.ToolCallEvent{}) {
			call := reflect.Indirect(reflect.ValueOf(payload)).Interface().(stream.ToolCallEvent)
			rawInput = bytes.Clone(call.Input)
			call.Input = nil
			payload = call
		}
		if err := mockJSONCopy(payload, data.Interface()); err != nil {
			return copy, fmt.Errorf("mock event %d: %w", i, err)
		}
		if typ == reflect.TypeOf(stream.ToolCallEvent{}) {
			data.Interface().(*stream.ToolCallEvent).Input = rawInput
		}
		if !pointer {
			data = data.Elem()
		}
		copy.Events[i].Data = data.Interface().(stream.EventData)
	}
	return copy, nil
}

// Requests returns independent snapshots in capture order. JSON values use
// encoding/json's decoded types (for example, numbers in any become float64).
// Contexts and tool Execute functions retain their identity.
func (m *MockModel) Requests() []stream.CallOptions {
	m.mu.Lock()
	defer m.mu.Unlock()
	requests := make([]stream.CallOptions, len(m.requests))
	for i, request := range m.requests {
		var err error
		requests[i], err = mockCopyRequest(request)
		if err != nil {
			panic(fmt.Sprintf("mock snapshot: %v", err))
		}
	}
	return requests
}

// ResetRequests clears captured requests without changing response rules.
func (m *MockModel) ResetRequests() {
	m.mu.Lock()
	m.requests = nil
	m.mu.Unlock()
}

func mockCopyRequest(request stream.CallOptions) (stream.CallOptions, error) {
	var copy stream.CallOptions
	if err := mockJSONCopy(request, &copy); err != nil {
		return copy, err
	}
	copy.AbortSignal = request.AbortSignal
	for i := range copy.Tools {
		copy.Tools[i].Execute = request.Tools[i].Execute
	}
	return copy, nil
}

func mockJSONCopy(src, dst any) error {
	data, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dst)
}

func mockLastUserText(messages []message.Message) (string, bool) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != message.RoleUser {
			continue
		}
		content := messages[i].Content
		if !content.IsMultiPart() {
			return content.Text, true
		}
		var text strings.Builder
		for _, part := range content.Parts {
			if p, ok := part.(message.TextPart); ok {
				text.WriteString(p.Text)
			}
		}
		return text.String(), true
	}
	return "", false
}
