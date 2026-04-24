package xai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// SSE fixture helpers ---------------------------------------------------------

// sseJSON formats a single SSE data line from a JSON-encodable value.
func sseJSON(v any) string {
	b, _ := json.Marshal(v)
	return fmt.Sprintf("data: %s\n\n", b)
}

// writeSSE serves a pre-built SSE body and sets the event-stream header.
func writeSSE(body string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(body))
	})
}

// captureBody wraps writeSSE but stores the request body into *dst first.
func captureBody(dst *map[string]any, body string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, dst)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(body))
	})
}

// Basic text-streaming fixture used by several tests.
func textStreamFixture(text string) string {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":         "resp_1",
			"created_at": 1,
			"model":      "grok-4",
		},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         map[string]any{"type": "message", "id": "msg_1", "role": "assistant"},
	}))
	for _, r := range text {
		sb.WriteString(sseJSON(map[string]any{
			"type":    "response.output_text.delta",
			"item_id": "msg_1",
			"delta":   string(r),
		}))
	}
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         map[string]any{"type": "message", "id": "msg_1", "role": "assistant"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":            "resp_1",
			"model":         "grok-4",
			"status":        "completed",
			"input_tokens":  10,
			"output_tokens": 20,
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 20,
			},
		},
	}))
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

// Helpers for collecting events ----------------------------------------------

type collected struct {
	events       []stream.Event
	text         string
	finishReason stream.FinishReason
	usage        stream.Usage
	finishMeta   map[string]any
	errors       []error
	toolCallIDs  []string
	toolInputs   map[string]string
	reasoningIDs []string
	reasoningEnd []stream.ReasoningEndEvent
	reasoningTxt map[string]string
}

func collect(events <-chan stream.Event) *collected {
	c := &collected{
		toolInputs:   make(map[string]string),
		reasoningTxt: make(map[string]string),
	}
	var toolBuf = make(map[string]*strings.Builder)
	for ev := range events {
		c.events = append(c.events, ev)
		switch e := ev.Data.(type) {
		case stream.TextDeltaEvent:
			c.text += e.Text
		case stream.ErrorEvent:
			c.errors = append(c.errors, e.Error)
		case stream.FinishEvent:
			c.finishReason = e.FinishReason
			c.usage = e.Usage
			c.finishMeta = e.ProviderMetadata
		case stream.ToolInputStartEvent:
			c.toolCallIDs = append(c.toolCallIDs, e.ID)
			toolBuf[e.ID] = &strings.Builder{}
		case stream.ToolInputDeltaEvent:
			if b, ok := toolBuf[e.ID]; ok {
				b.WriteString(e.Delta)
			}
		case stream.ToolInputEndEvent:
			if b, ok := toolBuf[e.ID]; ok {
				c.toolInputs[e.ID] = b.String()
			}
		case stream.ReasoningStartEvent:
			c.reasoningIDs = append(c.reasoningIDs, e.ID)
		case stream.ReasoningDeltaEvent:
			c.reasoningTxt[e.ID] += e.Text
		case stream.ReasoningEndEvent:
			c.reasoningEnd = append(c.reasoningEnd, e)
		}
	}
	return c
}

func newTestProvider(baseURL string) *Provider {
	return New(Options{APIKey: "test-key", BaseURL: baseURL})
}

// Tests -----------------------------------------------------------------------

func TestXaiResponses_ModelID(t *testing.T) {
	p := newTestProvider("http://localhost")
	m := p.Responses("grok-4")
	if m.ID() != "grok-4" {
		t.Errorf("ID = %s, want grok-4", m.ID())
	}
	if m.Provider() != "xai.responses" {
		t.Errorf("Provider = %s, want xai.responses", m.Provider())
	}
}

// Ensures grok-4* routes to Responses and grok-3* routes to Chat via compat.
func TestXaiResponses_ModelRouting(t *testing.T) {
	p := newTestProvider("http://localhost")

	responsesIDs := []string{
		"grok-4", "grok-4-0709", "grok-4-latest",
		"grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
		"grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning",
		"grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning",
		"grok-4.20-multi-agent-0309", "grok-code-fast-1",
	}
	for _, id := range responsesIDs {
		m := p.Model(id)
		if _, ok := m.(*XaiResponsesModel); !ok {
			t.Errorf("Model(%q) = %T, want *XaiResponsesModel", id, m)
		}
	}

	chatIDs := []string{"grok-3", "grok-3-mini", "grok-3-latest", "grok-beta"}
	for _, id := range chatIDs {
		m := p.Model(id)
		if _, ok := m.(*XaiResponsesModel); ok {
			t.Errorf("Model(%q) = *XaiResponsesModel, want openaicompat", id)
		}
	}
}

func TestXaiResponses_TextStreaming(t *testing.T) {
	server := httptest.NewServer(writeSSE(textStreamFixture("Hi there!")))
	defer server.Close()

	p := newTestProvider(server.URL)
	m := p.Responses("grok-4")

	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("Hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	c := collect(events)

	if c.text != "Hi there!" {
		t.Errorf("text = %q, want %q", c.text, "Hi there!")
	}
	if c.finishReason != stream.FinishReasonStop {
		t.Errorf("finishReason = %s, want stop", c.finishReason)
	}
	if c.usage.InputTotal() != 10 {
		t.Errorf("input tokens = %d, want 10", c.usage.InputTotal())
	}
	if c.usage.OutputTotal() != 20 {
		t.Errorf("output tokens = %d, want 20", c.usage.OutputTotal())
	}
}

// ReasoningSummary: summary_text.delta → reasoning-delta events.
func TestXaiResponses_ReasoningSummary(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         map[string]any{"type": "reasoning", "id": "rs_1"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.reasoning_summary_text.delta",
		"item_id": "rs_1",
		"delta":   "Thinking ",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.reasoning_summary_text.delta",
		"item_id": "rs_1",
		"delta":   "hard",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         map[string]any{"type": "reasoning", "id": "rs_1"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "resp_r", "model": "grok-4", "status": "completed"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	m := p.Responses("grok-4")

	events, _ := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	c := collect(events)

	if len(c.reasoningIDs) != 1 || c.reasoningIDs[0] != "rs_1" {
		t.Errorf("reasoning IDs = %v, want [rs_1]", c.reasoningIDs)
	}
	if c.reasoningTxt["rs_1"] != "Thinking hard" {
		t.Errorf("reasoning text = %q, want %q", c.reasoningTxt["rs_1"], "Thinking hard")
	}
	if len(c.reasoningEnd) != 1 {
		t.Errorf("expected 1 reasoning-end, got %d", len(c.reasoningEnd))
	}
}

// #8b3e72d: full reasoning_text.delta events.
func TestXaiResponses_ReasoningFullText(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         map[string]any{"type": "reasoning", "id": "rs_full"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.reasoning_text.delta",
		"item_id": "rs_full",
		"delta":   "Full ",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.reasoning_text.delta",
		"item_id": "rs_full",
		"delta":   "reasoning.",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         map[string]any{"type": "reasoning", "id": "rs_full"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "resp_r", "model": "grok-4", "status": "completed"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	c := collect(events)

	if c.reasoningTxt["rs_full"] != "Full reasoning." {
		t.Errorf("reasoning text = %q, want %q", c.reasoningTxt["rs_full"], "Full reasoning.")
	}
}

// #b937f3e + #58800f3: encrypted-only reasoning round-trip. reasoning-start
// must be emitted even without reasoning_summary_part.added, and
// reasoning-end must carry encrypted_content on providerMetadata.xai.
func TestXaiResponses_EncryptedReasoningRoundTrip(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_r", "model": "grok-4"}}))
	// No output_item.added for reasoning; no summary events. Encrypted-only.
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":              "reasoning",
			"id":                "rs_enc",
			"encrypted_content": "ENCRYPTED_BLOB",
		},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "resp_r", "model": "grok-4", "status": "completed"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: map[string]any{
			"store": false,
		},
	})
	c := collect(events)

	if len(c.reasoningIDs) != 1 {
		t.Fatalf("expected 1 reasoning-start, got %d", len(c.reasoningIDs))
	}
	if c.reasoningIDs[0] != "rs_enc" {
		t.Errorf("reasoning-start ID = %q, want rs_enc", c.reasoningIDs[0])
	}
	if len(c.reasoningEnd) != 1 {
		t.Fatalf("expected 1 reasoning-end, got %d", len(c.reasoningEnd))
	}
	xaiMeta, ok := c.reasoningEnd[0].ProviderMetadata["xai"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning-end providerMetadata.xai missing: %#v", c.reasoningEnd[0].ProviderMetadata)
	}
	if xaiMeta["reasoningEncryptedContent"] != "ENCRYPTED_BLOB" {
		t.Errorf("reasoningEncryptedContent = %v, want ENCRYPTED_BLOB", xaiMeta["reasoningEncryptedContent"])
	}
	if xaiMeta["itemId"] != "rs_enc" {
		t.Errorf("itemId = %v, want rs_enc", xaiMeta["itemId"])
	}
}

// #902e93b: function_call_arguments.delta → ToolInputDeltaEvent.
func TestXaiResponses_FunctionCallStreaming(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_t", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type":    "function_call",
			"id":      "fc_1",
			"call_id": "call_abc",
			"name":    "get_weather",
		},
	}))
	for _, part := range []string{`{"loc`, `ation":"SF"}`} {
		sb.WriteString(sseJSON(map[string]any{
			"type":         "response.function_call_arguments.delta",
			"output_index": 0,
			"item_id":      "fc_1",
			"delta":        part,
		}))
	}
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.function_call_arguments.done",
		"output_index": 0,
		"item_id":      "fc_1",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":      "function_call",
			"id":        "fc_1",
			"call_id":   "call_abc",
			"name":      "get_weather",
			"arguments": `{"location":"SF"}`,
		},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "resp_t", "model": "grok-4", "status": "completed"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("weather?")},
		Tools: []tool.Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	})
	c := collect(events)

	if len(c.toolCallIDs) != 1 || c.toolCallIDs[0] != "call_abc" {
		t.Errorf("tool call IDs = %v, want [call_abc]", c.toolCallIDs)
	}
	if c.toolInputs["call_abc"] != `{"location":"SF"}` {
		t.Errorf("streamed args = %q, want %q", c.toolInputs["call_abc"], `{"location":"SF"}`)
	}
	// ToolCallEvent should be emitted.
	var toolCalled bool
	for _, ev := range c.events {
		if tc, ok := ev.Data.(stream.ToolCallEvent); ok {
			toolCalled = true
			if tc.ToolName != "get_weather" {
				t.Errorf("tool name = %q, want get_weather", tc.ToolName)
			}
			if string(tc.Input) != `{"location":"SF"}` {
				t.Errorf("tool input = %q, want %q", tc.Input, `{"location":"SF"}`)
			}
		}
	}
	if !toolCalled {
		t.Error("expected ToolCallEvent")
	}
}

// #5d61547: tool-calls finish reason overrides "stop"/"completed" when a
// function_call was seen in the stream.
func TestXaiResponses_ToolFinishReason(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_t", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type":    "function_call",
			"id":      "fc_2",
			"call_id": "call_2",
			"name":    "do_it",
		},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":      "function_call",
			"id":        "fc_2",
			"call_id":   "call_2",
			"name":      "do_it",
			"arguments": `{}`,
		},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":     "resp_t",
			"model":  "grok-4",
			"status": "completed", // would normally map to stop
		},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if c.finishReason != stream.FinishReasonToolCalls {
		t.Errorf("finishReason = %s, want tool-calls", c.finishReason)
	}
}

// #c1cc97f: response.incomplete uses incomplete_details.reason.
func TestXaiResponses_ResponseIncomplete(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.incomplete",
		"response": map[string]any{
			"id":                 "r",
			"model":              "grok-4",
			"incomplete_details": map[string]any{"reason": "max_output_tokens"},
		},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if c.finishReason != stream.FinishReasonLength {
		t.Errorf("finishReason = %s, want length", c.finishReason)
	}
	if c.finishMeta == nil {
		t.Fatalf("expected providerMetadata on finish")
	}
	xaiMeta, _ := c.finishMeta["xai"].(map[string]any)
	if xaiMeta["rawFinishReason"] != "max_output_tokens" {
		t.Errorf("rawFinishReason = %v, want max_output_tokens", xaiMeta["rawFinishReason"])
	}
}

// #c1cc97f: response.failed with no reason falls back to FinishReasonError.
func TestXaiResponses_ResponseFailed(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.failed",
		"response": map[string]any{"id": "r", "model": "grok-4"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if c.finishReason != stream.FinishReasonError {
		t.Errorf("finishReason = %s, want error", c.finishReason)
	}
	xaiMeta, _ := c.finishMeta["xai"].(map[string]any)
	if xaiMeta["rawFinishReason"] != "error" {
		t.Errorf("rawFinishReason = %v, want error", xaiMeta["rawFinishReason"])
	}
}

// #72ebb54: mid-stream error events surface as ErrorEvent without
// aborting the stream; subsequent deltas + completion still flow through.
func TestXaiResponses_MidStreamError(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         map[string]any{"type": "message", "id": "msg_1", "role": "assistant"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.output_text.delta",
		"item_id": "msg_1",
		"delta":   "before-",
	}))
	// Mid-stream error event.
	sb.WriteString(sseJSON(map[string]any{
		"type": "error",
		"error": map[string]any{
			"code":    "rate_limited",
			"message": "slow down",
		},
	}))
	// Stream continues.
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.output_text.delta",
		"item_id": "msg_1",
		"delta":   "after",
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         map[string]any{"type": "message", "id": "msg_1", "role": "assistant"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "r", "model": "grok-4", "status": "completed"},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if len(c.errors) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(c.errors))
	}
	if !strings.Contains(c.errors[0].Error(), "rate_limited") {
		t.Errorf("error = %q, want contains rate_limited", c.errors[0])
	}
	if c.text != "before-after" {
		t.Errorf("text = %q, want 'before-after' (stream must continue after error)", c.text)
	}
	if c.finishReason != stream.FinishReasonStop {
		t.Errorf("finishReason = %s, want stop", c.finishReason)
	}
}

// #7ccb902: cached_tokens surfaces on Usage.InputTokens.CacheRead.
func TestXaiResponses_CachedTokens(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":     "r",
			"model":  "grok-4",
			"status": "completed",
			"usage": map[string]any{
				"input_tokens":         100,
				"output_tokens":        50,
				"input_tokens_details": map[string]any{"cached_tokens": 30},
			},
		},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if c.usage.InputTokens.CacheRead == nil {
		t.Fatal("CacheRead is nil, want 30")
	}
	if *c.usage.InputTokens.CacheRead != 30 {
		t.Errorf("CacheRead = %d, want 30", *c.usage.InputTokens.CacheRead)
	}
	if c.usage.InputTokens.NoCache == nil || *c.usage.InputTokens.NoCache != 70 {
		t.Errorf("NoCache = %v, want 70", c.usage.InputTokens.NoCache)
	}
	if c.usage.InputTotal() != 100 {
		t.Errorf("InputTotal = %d, want 100", c.usage.InputTotal())
	}
}

// #e1d5111: reasoning_tokens surfaces on Usage.OutputTokens.Reasoning and
// Text is populated as output_tokens - reasoning_tokens.
func TestXaiResponses_ReasoningTokens(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":     "r",
			"model":  "grok-4",
			"status": "completed",
			"usage": map[string]any{
				"input_tokens":          10,
				"output_tokens":         200,
				"output_tokens_details": map[string]any{"reasoning_tokens": 150},
			},
		},
	}))
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if c.usage.OutputTokens.Reasoning == nil || *c.usage.OutputTokens.Reasoning != 150 {
		t.Errorf("Reasoning tokens = %v, want 150", c.usage.OutputTokens.Reasoning)
	}
	if c.usage.OutputTokens.Text == nil || *c.usage.OutputTokens.Text != 50 {
		t.Errorf("Text tokens = %v, want 50", c.usage.OutputTokens.Text)
	}
}

// #de16a00: stream closes with no response.completed/incomplete/failed → a
// synthetic zero-filled Usage + other finish reason must still be emitted.
func TestXaiResponses_MissingUsageFallback(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(sseJSON(map[string]any{"type": "response.created", "response": map[string]any{"id": "r", "model": "grok-4"}}))
	sb.WriteString(sseJSON(map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         map[string]any{"type": "message", "id": "msg_1", "role": "assistant"},
	}))
	sb.WriteString(sseJSON(map[string]any{
		"type":    "response.output_text.delta",
		"item_id": "msg_1",
		"delta":   "partial",
	}))
	// No output_item.done, no response.completed — stream just ends.
	sb.WriteString("data: [DONE]\n\n")

	server := httptest.NewServer(writeSSE(sb.String()))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	// Finish event must still have been emitted.
	var finished bool
	for _, ev := range c.events {
		if ev.Type == stream.EventFinish {
			finished = true
		}
	}
	if !finished {
		t.Fatal("expected EventFinish even when provider never sent response.completed")
	}
	if c.finishReason != stream.FinishReasonOther {
		t.Errorf("finishReason = %s, want other", c.finishReason)
	}
	// Zero-filled totals.
	if c.usage.InputTotal() != 0 || c.usage.OutputTotal() != 0 {
		t.Errorf("usage = %+v, want zero totals", c.usage)
	}
}

// Verify the request body carries through reasoningEffort, logprobs,
// topLogprobs, store, previousResponseId, include options.
func TestXaiResponses_RequestBody(t *testing.T) {
	t.Run("reasoningEffort", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages:        []message.Message{message.NewUserMessage("x")},
			ProviderOptions: map[string]any{"reasoningEffort": "high"},
		})
		for range events {
		}

		reasoning, ok := captured["reasoning"].(map[string]any)
		if !ok {
			t.Fatalf("expected reasoning object, got %v", captured["reasoning"])
		}
		if reasoning["effort"] != "high" {
			t.Errorf("reasoning.effort = %v, want high", reasoning["effort"])
		}
	})

	t.Run("logprobs and topLogprobs", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages:        []message.Message{message.NewUserMessage("x")},
			ProviderOptions: map[string]any{"topLogprobs": float64(3)},
		})
		for range events {
		}

		if captured["logprobs"] != true {
			t.Errorf("logprobs = %v, want true when topLogprobs set", captured["logprobs"])
		}
		if captured["top_logprobs"] != float64(3) {
			t.Errorf("top_logprobs = %v, want 3", captured["top_logprobs"])
		}
	})

	t.Run("store=false adds reasoning.encrypted_content include and strips IDs", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("x")},
			ProviderOptions: map[string]any{
				"store": false,
			},
		})
		for range events {
		}

		if captured["store"] != false {
			t.Errorf("store = %v, want false", captured["store"])
		}
		includeArr, _ := captured["include"].([]any)
		foundEncrypted := false
		for _, v := range includeArr {
			if v == "reasoning.encrypted_content" {
				foundEncrypted = true
			}
		}
		if !foundEncrypted {
			t.Errorf("include missing reasoning.encrypted_content: %v", includeArr)
		}
	})

	t.Run("previousResponseId", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages:        []message.Message{message.NewUserMessage("x")},
			ProviderOptions: map[string]any{"previousResponseId": "resp_prev"},
		})
		for range events {
		}

		if captured["previous_response_id"] != "resp_prev" {
			t.Errorf("previous_response_id = %v, want resp_prev", captured["previous_response_id"])
		}
	})

	t.Run("include passthrough", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("x")},
			ProviderOptions: map[string]any{
				"include": []string{"file_search_call.results"},
			},
		})
		for range events {
		}

		includeArr, _ := captured["include"].([]any)
		if len(includeArr) != 1 || includeArr[0] != "file_search_call.results" {
			t.Errorf("include = %v, want [file_search_call.results]", includeArr)
		}
	})

	t.Run("model + input wired correctly", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
		defer server.Close()

		p := newTestProvider(server.URL)
		events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("be helpful"),
				message.NewUserMessage("x"),
			},
		})
		for range events {
		}

		if captured["model"] != "grok-4" {
			t.Errorf("model = %v, want grok-4", captured["model"])
		}
		if captured["stream"] != true {
			t.Errorf("stream = %v, want true", captured["stream"])
		}

		input, _ := captured["input"].([]any)
		if len(input) != 2 {
			t.Fatalf("input length = %d, want 2", len(input))
		}
		sysItem := input[0].(map[string]any)
		if sysItem["role"] != "system" {
			t.Errorf("system item role = %v, want system (xAI uses literal system role)", sysItem["role"])
		}
		userItem := input[1].(map[string]any)
		if userItem["role"] != "user" {
			t.Errorf("user item role = %v, want user", userItem["role"])
		}
	})
}

// Verify the Authorization header carries the API key.
func TestXaiResponses_Headers(t *testing.T) {
	var gotHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(textStreamFixture("ok")))
	}))
	defer server.Close()

	p := New(Options{
		APIKey:  "xai-key",
		BaseURL: server.URL,
		Headers: map[string]string{"X-Provider": "p"},
	})
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
		Headers:  map[string]string{"X-Request": "r"},
	})
	for range events {
	}

	if gotHeaders.Get("Authorization") != "Bearer xai-key" {
		t.Errorf("Authorization = %q, want 'Bearer xai-key'", gotHeaders.Get("Authorization"))
	}
	if gotHeaders.Get("X-Provider") != "p" {
		t.Errorf("X-Provider header missing")
	}
	if gotHeaders.Get("X-Request") != "r" {
		t.Errorf("X-Request header missing")
	}
}

// Verify HTTP error responses surface as ErrorEvent.
func TestXaiResponses_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"bad"}}`))
	}))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("x")},
	})
	c := collect(events)

	if len(c.errors) == 0 {
		t.Fatal("expected an error event")
	}
	if !strings.Contains(c.errors[0].Error(), "status 400") {
		t.Errorf("error = %q, want contains status 400", c.errors[0])
	}
}

// Verify mapXaiResponsesFinishReason covers the full vocabulary.
func TestXaiResponses_MapFinishReason(t *testing.T) {
	cases := []struct {
		reason          string
		hasFunctionCall bool
		want            stream.FinishReason
	}{
		{"stop", false, stream.FinishReasonStop},
		{"completed", false, stream.FinishReasonStop},
		{"stop", true, stream.FinishReasonToolCalls},
		{"completed", true, stream.FinishReasonToolCalls},
		{"length", false, stream.FinishReasonLength},
		{"max_output_tokens", false, stream.FinishReasonLength},
		{"tool_calls", false, stream.FinishReasonToolCalls},
		{"function_call", false, stream.FinishReasonToolCalls},
		{"content_filter", false, stream.FinishReasonContentFilter},
		{"", true, stream.FinishReasonToolCalls},
		{"", false, stream.FinishReasonOther},
		{"unknown-thing", false, stream.FinishReasonOther},
	}
	for _, tc := range cases {
		got := mapXaiResponsesFinishReason(tc.reason, tc.hasFunctionCall)
		if got != tc.want {
			t.Errorf("mapXaiResponsesFinishReason(%q, %v) = %s, want %s", tc.reason, tc.hasFunctionCall, got, tc.want)
		}
	}
}

// Usage-converter unit test. xAI reports cached_tokens separately from
// input_tokens when the API computes them differently; the converter
// must add them together in that case.
func TestXaiResponses_UsageConverter(t *testing.T) {
	t.Run("cached tokens subsumed by input_tokens", func(t *testing.T) {
		u := &responsesUsage{
			InputTokens:  100,
			OutputTokens: 50,
			InputTokensDetails: &struct {
				CachedTokens int `json:"cached_tokens,omitempty"`
			}{CachedTokens: 40},
		}
		got := convertXaiResponsesUsage(u)
		if *got.InputTokens.Total != 100 {
			t.Errorf("Total = %d, want 100", *got.InputTokens.Total)
		}
		if *got.InputTokens.NoCache != 60 {
			t.Errorf("NoCache = %d, want 60", *got.InputTokens.NoCache)
		}
		if *got.InputTokens.CacheRead != 40 {
			t.Errorf("CacheRead = %d, want 40", *got.InputTokens.CacheRead)
		}
	})

	t.Run("cached tokens reported separately", func(t *testing.T) {
		u := &responsesUsage{
			InputTokens:  60, // non-cached only
			OutputTokens: 50,
			InputTokensDetails: &struct {
				CachedTokens int `json:"cached_tokens,omitempty"`
			}{CachedTokens: 200},
		}
		got := convertXaiResponsesUsage(u)
		if *got.InputTokens.Total != 260 {
			t.Errorf("Total = %d, want 260 (60 + 200)", *got.InputTokens.Total)
		}
		if *got.InputTokens.NoCache != 60 {
			t.Errorf("NoCache = %d, want 60", *got.InputTokens.NoCache)
		}
		if *got.InputTokens.CacheRead != 200 {
			t.Errorf("CacheRead = %d, want 200", *got.InputTokens.CacheRead)
		}
	})

	t.Run("nil details → nil pointers for optional fields", func(t *testing.T) {
		u := &responsesUsage{InputTokens: 10, OutputTokens: 5}
		got := convertXaiResponsesUsage(u)
		if got.InputTokens.CacheRead != nil {
			t.Errorf("CacheRead = %v, want nil", got.InputTokens.CacheRead)
		}
		if got.OutputTokens.Reasoning != nil {
			t.Errorf("Reasoning = %v, want nil", got.OutputTokens.Reasoning)
		}
	})
}

// Verify convertToResponsesInput emits literal "system" role (no
// systemMessageMode branch), and preserves reasoning round-trip metadata.
func TestXaiResponses_ConvertToResponsesInput(t *testing.T) {
	t.Run("system role is literal", func(t *testing.T) {
		items := convertToResponsesInput([]message.Message{
			message.NewSystemMessage("sys"),
			message.NewUserMessage("hi"),
		})
		if len(items) != 2 {
			t.Fatalf("items length = %d, want 2", len(items))
		}
		if items[0].Role != "system" {
			t.Errorf("system role = %q, want system", items[0].Role)
		}
		b, _ := json.Marshal(items[0])
		if !strings.Contains(string(b), `"role":"system"`) {
			t.Errorf("marshaled system = %s, want role=system", string(b))
		}
	})

	t.Run("reasoning with encrypted content round-trips", func(t *testing.T) {
		items := convertToResponsesInput([]message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ReasoningPart{
							Text: "thinking",
							ProviderOptions: map[string]any{
								"xai": map[string]any{
									"itemId":                    "rs_1",
									"reasoningEncryptedContent": "ENC",
								},
							},
						},
					},
				},
			},
		})
		if len(items) != 1 {
			t.Fatalf("items length = %d, want 1", len(items))
		}
		if items[0].Type != "reasoning" {
			t.Errorf("type = %q, want reasoning", items[0].Type)
		}
		if items[0].ID != "rs_1" {
			t.Errorf("id = %q, want rs_1", items[0].ID)
		}
		if items[0].EncryptedContent != "ENC" {
			t.Errorf("encrypted = %q, want ENC", items[0].EncryptedContent)
		}
	})

	t.Run("tool call with default args when Input empty", func(t *testing.T) {
		items := convertToResponsesInput([]message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolCallPart{ID: "call_1", Name: "x", Input: nil},
					},
				},
			},
		})
		if items[0].Arguments != "{}" {
			t.Errorf("arguments = %q, want '{}'", items[0].Arguments)
		}
	})

	t.Run("tool-result mapped to function_call_output", func(t *testing.T) {
		items := convertToResponsesInput([]message.Message{
			{
				Role: message.RoleTool,
				Content: message.Content{
					Parts: []message.Part{
						message.ToolResultPart{ToolCallID: "call_1", ToolName: "x", Result: "got it"},
					},
				},
			},
		})
		if items[0].Type != "function_call_output" {
			t.Errorf("type = %q, want function_call_output", items[0].Type)
		}
		if items[0].CallID != "call_1" {
			t.Errorf("call_id = %q, want call_1", items[0].CallID)
		}
		if items[0].Output != "got it" {
			t.Errorf("output = %v, want 'got it'", items[0].Output)
		}
	})
}

// Reasoning parts with neither itemId nor encryptedContent must be dropped
// defensively when store=false (symmetric to OpenAI Responses behavior).
func TestXaiResponses_StoreFalseDropsReasoningWithoutEncrypted(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(captureBody(&captured, textStreamFixture("ok")))
	defer server.Close()

	p := newTestProvider(server.URL)
	events, _ := p.Responses("grok-4").Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{
					Parts: []message.Part{
						// No itemId, no encrypted — should be skipped by converter entirely.
						message.ReasoningPart{Text: "hmm"},
						// Kept: has encrypted_content.
						message.ReasoningPart{
							Text: "keepme",
							ProviderOptions: map[string]any{
								"xai": map[string]any{
									"reasoningEncryptedContent": "ENC",
								},
							},
						},
					},
				},
			},
			message.NewUserMessage("x"),
		},
		ProviderOptions: map[string]any{"store": false},
	})
	for range events {
	}

	input, _ := captured["input"].([]any)
	var reasoningItems int
	for _, v := range input {
		if item, ok := v.(map[string]any); ok && item["type"] == "reasoning" {
			reasoningItems++
			if item["encrypted_content"] != "ENC" {
				t.Errorf("kept reasoning has wrong encrypted_content: %v", item)
			}
			if _, hasID := item["id"]; hasID {
				if item["id"] != "" {
					t.Errorf("store=false must strip ids but found id=%v", item["id"])
				}
			}
		}
	}
	if reasoningItems != 1 {
		t.Errorf("reasoning items = %d, want 1 (the one with encrypted_content)", reasoningItems)
	}
}
