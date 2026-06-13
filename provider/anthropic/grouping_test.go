package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Anthropic strictly requires that every tool_result for a given assistant
// turn lives in ONE user message immediately following that assistant turn.
// goai used to emit one anthropic user message per goai tool message (and one
// per ToolResultPart), which Anthropic rejects with HTTP 400:
//
//	messages.N.content.K: unexpected tool_use_id found in tool_result blocks: <id>.
//	Each tool_result block must have a corresponding tool_use block in the
//	previous message.
//
// These tests mirror ai-sdk's groupIntoBlocks design
// (references/ai-sdk/packages/anthropic/src/convert-to-anthropic-prompt.ts:1088):
// consecutive user-role and tool-role goai messages must collapse into one
// anthropic user message; consecutive assistant-role goai messages must
// collapse into one anthropic assistant message.

// runBuildRequestBody returns the parsed request body produced by the same
// path the live Stream call uses. Local helper so failures point at the
// public BuildRequestBody surface (not internal helpers we may rename).
func runBuildRequestBody(t *testing.T, opts *stream.CallOptions) map[string]any {
	t.Helper()
	body, _, _, err := BuildRequestBody(Config{}, "claude-test", opts)
	if err != nil {
		t.Fatalf("BuildRequestBody: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	return parsed
}

func TestGrouping_TwoConsecutiveToolMessages(t *testing.T) {
	// Bug repro. Two RoleTool messages each with one ToolResultPart following
	// a single assistant message that emitted two parallel tool_use blocks.
	// Wire shape must be:
	//   user(text)
	//   assistant(tool_use_a, tool_use_b)
	//   user(tool_result_a, tool_result_b)
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("run both tools"),
			{
				Role: message.RoleAssistant,
				Content: message.Content{Parts: []message.Part{
					message.ToolCallPart{ID: "toolu_a", Name: "search", Input: json.RawMessage(`{"q":"a"}`)},
					message.ToolCallPart{ID: "toolu_b", Name: "search", Input: json.RawMessage(`{"q":"b"}`)},
				}},
			},
			{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "toolu_a", ToolName: "search", Output: message.TextOutput{Value: "result A"}},
				}},
			},
			{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "toolu_b", ToolName: "search", Output: message.TextOutput{Value: "result B"}},
				}},
			},
		},
	})

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (user, assistant, user), got %d: %v", len(msgs), msgs)
	}
	last, _ := msgs[2].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("expected role=user, got %v", last["role"])
	}
	content, _ := last["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 tool_result blocks merged into one user message, got %d: %v", len(content), content)
	}
	for i, want := range []string{"toolu_a", "toolu_b"} {
		blk, _ := content[i].(map[string]any)
		if blk["type"] != "tool_result" {
			t.Errorf("content[%d] type=%v, want tool_result", i, blk["type"])
		}
		if blk["tool_use_id"] != want {
			t.Errorf("content[%d] tool_use_id=%v, want %s", i, blk["tool_use_id"], want)
		}
	}
}

func TestGrouping_OneToolMessageMultipleResults(t *testing.T) {
	// One RoleTool message carrying two ToolResultParts must produce one
	// anthropic user message with two tool_result blocks. The pre-fix
	// convertToolMessages emitted one anthropic message per ToolResultPart.
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("run both"),
			{
				Role: message.RoleAssistant,
				Content: message.Content{Parts: []message.Part{
					message.ToolCallPart{ID: "toolu_a", Name: "search", Input: json.RawMessage(`{}`)},
					message.ToolCallPart{ID: "toolu_b", Name: "search", Input: json.RawMessage(`{}`)},
				}},
			},
			{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "toolu_a", ToolName: "search", Output: message.TextOutput{Value: "A"}},
					message.ToolResultPart{ToolCallID: "toolu_b", ToolName: "search", Output: message.TextOutput{Value: "B"}},
				}},
			},
		},
	})

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	last, _ := msgs[2].(map[string]any)
	content, _ := last["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 tool_result blocks in one user message, got %d", len(content))
	}
}

func TestGrouping_UserPlusToolMergeIntoOneMessage(t *testing.T) {
	// A RoleUser message followed by a RoleTool message must produce one
	// anthropic user message containing the user's text part plus the
	// tool_result blocks (ai-sdk parity — both share the user block).
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{
			{
				Role: message.RoleAssistant,
				Content: message.Content{Parts: []message.Part{
					message.ToolCallPart{ID: "toolu_a", Name: "lookup", Input: json.RawMessage(`{}`)},
				}},
			},
			{
				Role: message.RoleTool,
				Content: message.Content{Parts: []message.Part{
					message.ToolResultPart{ToolCallID: "toolu_a", ToolName: "lookup", Output: message.TextOutput{Value: "ok"}},
				}},
			},
			message.NewUserMessage("now do the next step"),
		},
	})

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (assistant, user), got %d: %v", len(msgs), msgs)
	}
	last, _ := msgs[1].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("expected role=user, got %v", last["role"])
	}
	content, _ := last["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 blocks (tool_result + text), got %d: %v", len(content), content)
	}
	first, _ := content[0].(map[string]any)
	if first["type"] != "tool_result" {
		t.Errorf("content[0] type=%v, want tool_result", first["type"])
	}
	second, _ := content[1].(map[string]any)
	if second["type"] != "text" {
		t.Errorf("content[1] type=%v, want text", second["type"])
	}
	if second["text"] != "now do the next step" {
		t.Errorf("content[1] text=%v, want 'now do the next step'", second["text"])
	}
}

func TestGrouping_TwoConsecutiveAssistantMessages(t *testing.T) {
	// Two consecutive RoleAssistant messages must collapse into a single
	// anthropic assistant message with merged content blocks. Mirrors
	// ai-sdk's "combines multiple assistant messages in this block" comment
	// at convert-to-anthropic-prompt.ts:453.
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("hi"),
			{
				Role:    message.RoleAssistant,
				Content: message.Content{Text: "first chunk"},
			},
			{
				Role: message.RoleAssistant,
				Content: message.Content{Parts: []message.Part{
					message.ToolCallPart{ID: "toolu_x", Name: "tool", Input: json.RawMessage(`{}`)},
				}},
			},
		},
	})

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (user, assistant), got %d: %v", len(msgs), msgs)
	}
	asst, _ := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("expected role=assistant, got %v", asst["role"])
	}
	content, _ := asst["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 merged content blocks (text + tool_use), got %d: %v", len(content), content)
	}
	first, _ := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "first chunk" {
		t.Errorf("content[0]=%v, want type=text text='first chunk'", first)
	}
	second, _ := content[1].(map[string]any)
	if second["type"] != "tool_use" || second["id"] != "toolu_x" {
		t.Errorf("content[1]=%v, want type=tool_use id=toolu_x", second)
	}
}

// Sanity check — single-message paths still emit one anthropic message each.
// Guards against an over-eager merge that would break the common case.
func TestGrouping_AlternatingShapeUnchanged(t *testing.T) {
	body := runBuildRequestBody(t, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("hello"),
			{Role: message.RoleAssistant, Content: message.Content{Text: "hi"}},
			message.NewUserMessage("how are you"),
			{Role: message.RoleAssistant, Content: message.Content{Text: "fine"}},
		},
	})
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("alternating user/assistant/user/assistant should produce 4 messages, got %d", len(msgs))
	}
}
