package openaicompat

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/message"
)

// helper: assistant message carrying one or more tool calls.
func asstWithCalls(calls ...message.ToolCallPart) message.Message {
	parts := make([]message.Part, len(calls))
	for i, c := range calls {
		parts[i] = c
	}
	return message.Message{Role: message.RoleAssistant, Content: message.Content{Parts: parts}}
}

func toolResultMsg(callID, name, out string) message.Message {
	return message.NewToolMessage(callID, name, message.TextOutput{Value: out})
}

// roles flattens a message slice to its ordered roles for compact assertions.
func roles(msgs []message.Message) []string {
	r := make([]string, len(msgs))
	for i, m := range msgs {
		r[i] = string(m.Role)
	}
	return r
}

func TestPairToolResults_AllAnswered_ReturnsInputUnchanged(t *testing.T) {
	in := []message.Message{
		message.NewUserMessage("hi"),
		asstWithCalls(
			message.ToolCallPart{ID: "a", Name: "read", Input: json.RawMessage(`{}`)},
			message.ToolCallPart{ID: "b", Name: "grep", Input: json.RawMessage(`{}`)},
		),
		toolResultMsg("a", "read", "ra"),
		toolResultMsg("b", "grep", "rb"),
	}
	out := pairToolResults(in)
	// Nothing missing -> same backing slice returned.
	if len(out) != len(in) {
		t.Fatalf("len = %d, want %d", len(out), len(in))
	}
	want := []string{"user", "assistant", "tool", "tool"}
	if got := roles(out); !equalStrings(got, want) {
		t.Errorf("roles = %v, want %v", got, want)
	}
}

func TestPairToolResults_SynthesizesMissingResult(t *testing.T) {
	// Parallel calls a + b; only a is answered (b was NoExecute / dropped).
	in := []message.Message{
		message.NewUserMessage("hi"),
		asstWithCalls(
			message.ToolCallPart{ID: "a", Name: "read", Input: json.RawMessage(`{}`)},
			message.ToolCallPart{ID: "b", Name: "ask", Input: json.RawMessage(`{}`)},
		),
		toolResultMsg("a", "read", "ra"),
		message.NewUserMessage("next"),
	}
	out := pairToolResults(in)

	want := []string{"user", "assistant", "tool", "tool", "user"}
	if got := roles(out); !equalStrings(got, want) {
		t.Fatalf("roles = %v, want %v", got, want)
	}
	// The synthetic tool message must answer call "b" and sit before the next user turn.
	synth := out[3]
	tr, ok := synth.Content.Parts[0].(message.ToolResultPart)
	if !ok {
		t.Fatalf("expected ToolResultPart, got %T", synth.Content.Parts[0])
	}
	if tr.ToolCallID != "b" {
		t.Errorf("synthetic answers %q, want b", tr.ToolCallID)
	}
	if _, ok := tr.Output.(message.ErrorTextOutput); !ok {
		t.Errorf("synthetic output type = %T, want ErrorTextOutput", tr.Output)
	}
}

func TestPairToolResults_DanglingFinalAssistant(t *testing.T) {
	// Assistant tool_call is the last message (suspended/poisoned history).
	in := []message.Message{
		message.NewUserMessage("hi"),
		asstWithCalls(message.ToolCallPart{ID: "x", Name: "bash", Input: json.RawMessage(`{}`)}),
	}
	out := pairToolResults(in)
	want := []string{"user", "assistant", "tool"}
	if got := roles(out); !equalStrings(got, want) {
		t.Fatalf("roles = %v, want %v", got, want)
	}
	tr := out[2].Content.Parts[0].(message.ToolResultPart)
	if tr.ToolCallID != "x" {
		t.Errorf("synthetic answers %q, want x", tr.ToolCallID)
	}
}

func TestPairToolResults_PlainConversationUntouched(t *testing.T) {
	in := []message.Message{
		message.NewSystemMessage("sys"),
		message.NewUserMessage("hi"),
		message.NewAssistantMessage("hello"),
	}
	out := pairToolResults(in)
	if len(out) != len(in) {
		t.Fatalf("len = %d, want %d", len(out), len(in))
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
