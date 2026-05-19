package deepseek

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Mirrors ai-sdk's convert-to-deepseek-chat-messages.test.ts (PR #14739).
// DeepSeek's two model families have opposite reasoning_content rules:
// V4 must echo it; R1 must strip prior turns. Plain deepseek-chat /
// deepseek-coder don't carry the field at all.

func assistantMsg(parts ...message.Part) message.Message {
	return message.Message{
		Role:    message.RoleAssistant,
		Content: message.Content{Parts: parts},
	}
}

// Helper: pull out only assistant messages from converter output.
func assistantOnly(t *testing.T, out []any) []map[string]any {
	t.Helper()
	var result []map[string]any
	for _, m := range out {
		am, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", m)
		}
		if am["role"] == "assistant" {
			result = append(result, am)
		}
	}
	return result
}

func TestDeepSeekConvertMessages_V4_PreservesPriorReasoning(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("turn 1"),
		assistantMsg(
			message.ReasoningPart{Text: "thought 1"},
			message.TextPart{Text: "answer 1"},
		),
		message.NewUserMessage("turn 2"),
	}

	out, err := convertMessages("deepseek-v4-pro", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	asst := assistantOnly(t, out)
	if len(asst) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(asst))
	}
	if asst[0]["reasoning_content"] != "thought 1" {
		t.Errorf("V4 should preserve prior reasoning_content, got %v", asst[0]["reasoning_content"])
	}
	if asst[0]["content"] != "answer 1" {
		t.Errorf("content = %v, want answer 1", asst[0]["content"])
	}
}

func TestDeepSeekConvertMessages_V4_BackfillsEmptyReasoning(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("turn 1"),
		assistantMsg(message.TextPart{Text: "answer 1"}),
		message.NewUserMessage("turn 2"),
	}

	out, err := convertMessages("deepseek-v4", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	asst := assistantOnly(t, out)
	if len(asst) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(asst))
	}
	rc, has := asst[0]["reasoning_content"]
	if !has {
		t.Fatalf("V4 must back-fill reasoning_content even when source has no reasoning part; got %#v", asst[0])
	}
	if rc != "" {
		t.Errorf("V4 backfill should be empty string, got %v", rc)
	}
}

func TestDeepSeekConvertMessages_R1_StripsPriorReasoning(t *testing.T) {
	// R1 (deepseek-reasoner): prior-turn reasoning must be stripped. The
	// strip rule applies when the assistant turn appears at index <= the
	// index of the last user message — i.e. it's already been replied to.
	msgs := []message.Message{
		message.NewUserMessage("turn 1"),
		assistantMsg(
			message.ReasoningPart{Text: "old thought"},
			message.TextPart{Text: "answer 1"},
		),
		message.NewUserMessage("turn 2"),
	}

	out, err := convertMessages("deepseek-reasoner", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	asst := assistantOnly(t, out)
	if len(asst) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(asst))
	}
	if _, has := asst[0]["reasoning_content"]; has {
		t.Errorf("R1 must strip prior-turn reasoning_content, got %v", asst[0]["reasoning_content"])
	}
	if asst[0]["content"] != "answer 1" {
		t.Errorf("content = %v, want answer 1", asst[0]["content"])
	}
}

func TestDeepSeekConvertMessages_R1_KeepsReasoningAfterLastUser(t *testing.T) {
	// An assistant message that comes after the last user message keeps
	// its reasoning (used for in-flight reasoning during a multi-step
	// loop within a single turn).
	msgs := []message.Message{
		message.NewUserMessage("turn 1"),
		assistantMsg(
			message.ReasoningPart{Text: "current thought"},
			message.TextPart{Text: "answer"},
		),
	}

	out, err := convertMessages("deepseek-reasoner", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	asst := assistantOnly(t, out)
	if len(asst) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(asst))
	}
	if asst[0]["reasoning_content"] != "current thought" {
		t.Errorf("R1 should keep current-turn reasoning, got %v", asst[0]["reasoning_content"])
	}
}

func TestDeepSeekConvertMessages_PlainChat_OmitsReasoningField(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("hi"),
		assistantMsg(
			message.ReasoningPart{Text: "thought"},
			message.TextPart{Text: "hi back"},
		),
		message.NewUserMessage("again"),
	}

	for _, modelID := range []string{"deepseek-chat", "deepseek-coder"} {
		t.Run(modelID, func(t *testing.T) {
			out, err := convertMessages(modelID, msgs)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			asst := assistantOnly(t, out)
			if len(asst) != 1 {
				t.Fatalf("expected 1 assistant message, got %d", len(asst))
			}
			// deepseek-chat / -coder must drop prior reasoning by the R1
			// strip rule, AND must not back-fill (V4-only behavior).
			if _, has := asst[0]["reasoning_content"]; has {
				t.Errorf("%s must omit reasoning_content entirely, got %v", modelID, asst[0]["reasoning_content"])
			}
		})
	}
}

func TestDeepSeekConvertMessages_PreservesToolCallsAndResults(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("call weather"),
		assistantMsg(
			message.ToolCallPart{ID: "call_1", Name: "weather", Input: json.RawMessage(`{"city":"SF"}`)},
		),
		{
			Role: message.RoleTool,
			Content: message.Content{Parts: []message.Part{
				message.ToolResultPart{ToolCallID: "call_1", ToolName: "weather", Output: message.TextOutput{Value: "sunny"}},
			}},
		},
	}

	out, err := convertMessages("deepseek-chat", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 wire messages, got %d (%+v)", len(out), out)
	}

	asst, _ := out[1].(map[string]any)
	tc, _ := asst["tool_calls"].([]map[string]any)
	if len(tc) != 1 || tc[0]["id"] != "call_1" {
		t.Errorf("tool_calls not preserved: %+v", asst["tool_calls"])
	}

	toolMsg, _ := out[2].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "call_1" || toolMsg["content"] != "sunny" {
		t.Errorf("tool result not preserved: %+v", toolMsg)
	}
}

// End-to-end wire-shape check: the MessageConverter hook actually runs and
// reasoning_content reaches the HTTP body.
func TestDeepSeekModel_V4ReasoningContentOnWire(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.Model("deepseek-v4-pro")

	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("turn 1"),
			assistantMsg(
				message.ReasoningPart{Text: "I should check"},
				message.TextPart{Text: "ok"},
			),
			message.NewUserMessage("turn 2"),
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range events {
	}

	wireMessages, ok := captured["messages"].([]any)
	if !ok {
		t.Fatalf("expected messages on wire, got %T", captured["messages"])
	}
	var asst map[string]any
	for _, raw := range wireMessages {
		if m, ok := raw.(map[string]any); ok && m["role"] == "assistant" {
			asst = m
			break
		}
	}
	if asst == nil {
		t.Fatalf("no assistant message found on wire: %v", wireMessages)
	}
	if asst["reasoning_content"] != "I should check" {
		t.Errorf("wire reasoning_content = %v, want \"I should check\"", asst["reasoning_content"])
	}
}

// Round-trip JSON marshal sanity check — convertMessages output must be
// directly marshalable.
func TestDeepSeekConvertMessages_JSONRoundtrip(t *testing.T) {
	msgs := []message.Message{
		message.NewUserMessage("hi"),
		assistantMsg(message.TextPart{Text: "ok"}),
	}
	out, err := convertMessages("deepseek-v4", msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back []map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := []map[string]any{
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "ok", "reasoning_content": ""},
	}
	if !reflect.DeepEqual(back, want) {
		t.Errorf("roundtrip mismatch:\n got %#v\nwant %#v", back, want)
	}
}
