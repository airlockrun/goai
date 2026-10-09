package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/airlockrun/goai"
	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
)

type failingReadCloser struct {
	err error
}

func (r *failingReadCloser) Read([]byte) (int, error) { return 0, r.err }
func (r *failingReadCloser) Close() error             { return nil }

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestStreamReadErrorIsRetryable(t *testing.T) {
	readErr := errors.New("connection reset")
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       &failingReadCloser{err: readErr},
			Request:    req,
		}, nil
	})}

	events, err := Model("", Options{BaseURL: "http://proxy.test", Client: client}).Stream(context.Background(), &stream.CallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var eventErr error
	for event := range events {
		if event.Type == stream.EventError {
			eventErr = event.Data.(stream.ErrorEvent).Error
		}
		if event.Type == stream.EventFinish || event.Type == stream.EventFinishStep {
			t.Fatal("unexpected finish after stream read error")
		}
	}
	var apiErr *goaierrors.APICallError
	if !errors.Is(eventErr, readErr) || !errors.As(eventErr, &apiErr) || !apiErr.IsRetryable {
		t.Fatalf("error = %v, want retryable APICallError wrapping read error", eventErr)
	}
}

func TestStreamRetriesRetryableSetupStatus(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "upstream unavailable", http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"type\":\"text-delta\",\"data\":{\"text\":\"ok\"}}\n{\"type\":\"finish\",\"data\":{\"finishReason\":\"stop\"}}\n"))
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, Messages: []message.Message{message.NewUserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for event := range result.FullStream {
		if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
			text += delta.Text
		}
		if event.Type == stream.EventError {
			t.Fatalf("unexpected stream error: %v", event.Data)
		}
	}
	if text != "ok" {
		t.Fatalf("text = %q, want ok", text)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2", got)
	}
}

func TestStreamDoesNotRetryBodyError(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"type\":\"error\",\"data\":{\"error\":\"connection reset\"}}\n"))
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	events, err := model.Stream(context.Background(), &stream.CallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var sawError bool
	for event := range events {
		sawError = sawError || event.Type == stream.EventError
	}
	if !sawError {
		t.Fatal("expected body stream error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamTextExplicitZeroDisablesProxySetupRetries(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, MaxRetries: 0, MaxRetriesSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.Text(); err == nil {
		t.Fatal("Text() error = nil, want setup error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamTextHonorsNonRetryableAirlockSetupError(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Airlock-LLM-Retryable", "false")
		http.Error(w, "invalid provider options", http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.Text(); err == nil {
		t.Fatal("Text() error = nil, want setup error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamBoundsSetupErrorBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, strings.Repeat("x", maxErrorResponseBodyBytes*2), http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, MaxRetriesSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = result.Text()
	var apiErr *goaierrors.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Text() error = %v, want APICallError", err)
	}
	if len(apiErr.ResponseBody) > maxErrorResponseBodyBytes+len("... (truncated)") {
		t.Fatalf("response body length = %d, want bounded", len(apiErr.ResponseBody))
	}
	if !strings.HasSuffix(apiErr.ResponseBody, "... (truncated)") {
		t.Fatalf("response body was not marked truncated: %q", apiErr.ResponseBody[len(apiErr.ResponseBody)-32:])
	}
}

// proxyEventFixture sends canonical model events through the SDK's type/data
// NDJSON envelope and the real proxy HTTP decoder. SDK's internal serializer
// cannot be imported by GoAI; its error payload is represented as a string here.
func proxyEventFixture(t *testing.T, script []stream.Event) (stream.Model, *testutil.MockModel) {
	t.Helper()
	backend, err := testutil.NewMockModel(testutil.MockConfig{ID: "fixture", Default: &testutil.MockResponse{Events: script}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request proxyRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		var options stream.CallOptions
		if err := json.Unmarshal(request.Options, &options); err != nil {
			t.Error(err)
			http.Error(w, "bad options", http.StatusBadRequest)
			return
		}
		events, err := backend.Stream(r.Context(), &options)
		if err != nil {
			t.Error(err)
			http.Error(w, "model error", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		encoder := json.NewEncoder(w)
		for event := range events {
			data := any(event.Data)
			if failure, ok := event.Data.(stream.ErrorEvent); ok {
				data = struct {
					Error string `json:"error"`
				}{failure.Error.Error()}
			}
			if err := encoder.Encode(struct {
				Type string `json:"type"`
				Data any    `json:"data,omitempty"`
			}{string(event.Type), data}); err != nil {
				t.Error(err)
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return Model("fixture", Options{BaseURL: server.URL, Client: server.Client()}), backend
}

func TestStreamTextRawSourcesAndWarnings(t *testing.T) {
	warnings := []stream.Warning{
		stream.UnsupportedWarning("temperature", "ignored"),
		stream.CompatibilityWarning("topP", "clamped"),
		stream.OtherWarning("fixture warning"),
	}
	sources := []stream.SourceEvent{
		{SourceType: stream.SourceTypeURL, ID: "web", URL: "https://example.test/page", Title: "Page", ProviderMetadata: map[string]any{"citation": "web-1"}},
		{SourceType: stream.SourceTypeDocument, ID: "doc", MediaType: "application/pdf", Title: "Document", Filename: "report.pdf", ProviderMetadata: map[string]any{"file_id": "file-1"}},
	}
	raw := []stream.RawChunkEvent{{RawValue: `{"delta":"answer"}`}, {RawValue: map[string]any{"delta": "answer", "index": float64(1)}}}
	model, backend := proxyEventFixture(t, []stream.Event{
		{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}},
		{Type: stream.EventRawChunk, Data: raw[0]},
		{Type: stream.EventRawChunk, Data: raw[1]},
		{Type: stream.EventSource, Data: sources[0]},
		{Type: stream.EventSource, Data: sources[1]},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "answer"}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop, Usage: stream.UsageFrom(3, 2)}},
	})
	result, err := goai.StreamText(t.Context(), stream.Input{Model: model, IncludeRawChunks: true, Messages: []message.Message{message.NewUserMessage("question")}})
	if err != nil {
		t.Fatal(err)
	}
	var gotWarnings []stream.Warning
	var gotRaw []stream.RawChunkEvent
	var gotSources []stream.SourceEvent
	for event := range result.FullStream {
		switch data := event.Data.(type) {
		case stream.StartEvent:
			gotWarnings = append(gotWarnings, data.Warnings...)
		case stream.RawChunkEvent:
			gotRaw = append(gotRaw, data)
		case stream.SourceEvent:
			gotSources = append(gotSources, data)
		case stream.ErrorEvent:
			t.Errorf("stream error: %v", data.Error)
		}
	}
	if !reflect.DeepEqual(gotWarnings, warnings) || !reflect.DeepEqual(gotRaw, raw) || !reflect.DeepEqual(gotSources, sources) {
		t.Fatalf("warnings=%+v raw=%+v sources=%+v", gotWarnings, gotRaw, gotSources)
	}
	if !reflect.DeepEqual(result.Sources(), sources) {
		t.Fatalf("collected sources = %+v", result.Sources())
	}
	if text, err := result.Text(); text != "answer" || err != nil {
		t.Fatalf("text = %q, %v", text, err)
	}
	if reason, err := result.FinishReason(); reason != stream.FinishReasonStop || err != nil {
		t.Fatalf("finish = %q, %v", reason, err)
	}
	if usage, err := result.Usage(); usage.InputTotal() != 3 || usage.OutputTotal() != 2 || err != nil {
		t.Fatalf("usage = %+v, %v", usage, err)
	}
	requests := backend.Requests()
	if len(requests) != 1 || !requests[0].IncludeRawChunks {
		t.Fatalf("requests = %+v", requests)
	}
}

func TestStreamCanonicalEventUnion(t *testing.T) {
	metadata := map[string]any{"provider": "fixture"}
	input := json.RawMessage(`{"query":"test"}`)
	script := []stream.Event{
		{Type: stream.EventStart, Data: stream.StartEvent{Warnings: []stream.Warning{stream.OtherWarning("warning")}}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{ProviderMetadata: metadata}},
		{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "text", ProviderMetadata: metadata}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{ProviderMetadata: metadata}},
		{Type: stream.EventToolInputStart, Data: stream.ToolInputStartEvent{ID: "call", ToolName: "lookup", ProviderExecuted: true}},
		{Type: stream.EventToolInputDelta, Data: stream.ToolInputDeltaEvent{ID: "call", Delta: string(input)}},
		{Type: stream.EventToolInputEnd, Data: stream.ToolInputEndEvent{ID: "call"}},
		{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "call", ToolName: "lookup", Input: input, ProviderExecuted: true, ProviderMetadata: metadata}},
		{Type: stream.EventToolResult, Data: stream.ToolResultEvent{ToolCallID: "call", ToolName: "lookup", Input: input, Output: message.JSONOutput{Value: map[string]any{"found": true}}, ProviderExecuted: true, ProviderMetadata: metadata, Title: "Found", Metadata: metadata}},
		{Type: stream.EventToolError, Data: stream.ToolErrorEvent{ToolCallID: "failed", ToolName: "lookup", Input: input, Output: message.ErrorTextOutput{Value: "failed"}, ProviderExecuted: true, ProviderMetadata: metadata}},
		{Type: stream.EventToolOutputDenied, Data: stream.ToolOutputDeniedEvent{ToolCallID: "denied", ToolName: "lookup", Input: input, Reason: "denied"}},
		{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "r", ProviderMetadata: metadata}},
		{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "r", Text: "thinking", ProviderMetadata: metadata}},
		{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "r", ProviderMetadata: metadata}},
		{Type: stream.EventStartStep, Data: stream.StartStepEvent{}},
		{Type: stream.EventFinishStep, Data: stream.FinishStepEvent{FinishReason: stream.FinishReasonToolCalls, Usage: stream.UsageFrom(2, 1), ProviderMetadata: metadata}},
		{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: "raw payload"}},
		{Type: stream.EventSource, Data: stream.SourceEvent{SourceType: stream.SourceTypeURL, ID: "source", URL: "https://example.test", ProviderMetadata: metadata}},
		{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop, Usage: stream.UsageFrom(4, 2), ProviderMetadata: metadata}},
		{Type: stream.EventError, Data: stream.ErrorEvent{Error: errors.New("wire error")}},
	}
	model, _ := proxyEventFixture(t, script)
	events, err := model.Stream(t.Context(), &stream.CallOptions{IncludeRawChunks: true})
	if err != nil {
		t.Fatal(err)
	}
	var got []stream.Event
	for event := range events {
		got = append(got, event)
	}
	if len(got) != len(script) {
		t.Fatalf("got %d events, want %d: %+v", len(got), len(script), got)
	}
	for i, want := range script {
		if want.Type == stream.EventError {
			if got[i].Type != want.Type || got[i].Data.(stream.ErrorEvent).Error.Error() != want.Data.(stream.ErrorEvent).Error.Error() {
				t.Fatalf("error event = %+v", got[i])
			}
			continue
		}
		if !reflect.DeepEqual(got[i], want) {
			t.Errorf("event %q = %+v, want %+v", want.Type, got[i], want)
		}
	}
}

func TestStreamRejectsUnsupportedAndMalformedEvents(t *testing.T) {
	for _, tt := range []struct{ name, line, want string }{
		{"unsupported", `{"type":"future-event","data":{}}`, `unknown event type "future-event"`},
		{"missing discriminator", `{"data":{}}`, `unknown event type ""`},
		{"invalid JSON", `{"type":`, "parse event"},
		{"invalid raw payload", `{"type":"raw","data":"wrong"}`, `parse "raw" event data`},
		{"invalid source payload", `{"type":"source","data":{"title":42}}`, `parse "source" event data`},
		{"invalid warnings", `{"type":"start","data":{"warnings":"wrong"}}`, `parse "start" event data`},
		{"unknown output union", `{"type":"tool-result","data":{"output":{"type":"unsupported"}}}`, `parse "tool-result" event data`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				w.Header().Set("Content-Type", "application/x-ndjson")
				_, _ = w.Write([]byte("{\"type\":\"text-delta\",\"data\":{\"text\":\"partial\"}}\n" + tt.line + "\n{\"type\":\"finish\",\"data\":{\"finishReason\":\"stop\"}}\n"))
			}))
			defer server.Close()
			result, err := goai.StreamText(t.Context(), stream.Input{Model: Model("", Options{BaseURL: server.URL, Client: server.Client()})})
			if err != nil {
				t.Fatal(err)
			}
			var sawError, sawFinish bool
			for event := range result.FullStream {
				sawError = sawError || event.Type == stream.EventError
				sawFinish = sawFinish || event.Type == stream.EventFinish || event.Type == stream.EventFinishStep
			}
			text, err := result.Text()
			if text != "partial" || err == nil || !strings.Contains(err.Error(), tt.want) || !sawError || sawFinish {
				t.Fatalf("text=%q error=%v error event=%v finish=%v", text, err, sawError, sawFinish)
			}
			if attempts.Load() != 1 {
				t.Fatalf("attempts = %d, want no retry after partial output", attempts.Load())
			}
		})
	}
}

func TestStreamMarkerEventsWithoutData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"type\":\"start\"}\n{\"type\":\"start-step\"}\n{\"type\":\"text-start\"}\n{\"type\":\"text-end\"}\n"))
	}))
	defer server.Close()
	ch, err := Model("", Options{BaseURL: server.URL, Client: server.Client()}).Stream(t.Context(), &stream.CallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var got []stream.Event
	for event := range ch {
		got = append(got, event)
	}
	want := []stream.Event{
		{Type: stream.EventStart, Data: stream.StartEvent{}},
		{Type: stream.EventStartStep, Data: stream.StartStepEvent{}},
		{Type: stream.EventTextStart, Data: stream.TextStartEvent{}},
		{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("markers = %+v, want %+v", got, want)
	}
}
