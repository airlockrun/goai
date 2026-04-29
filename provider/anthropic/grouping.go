package anthropic

import "github.com/airlockrun/goai/message"

// messageBlockType is one of "system" / "assistant" / "user".
type messageBlockType string

const (
	blockSystem    messageBlockType = "system"
	blockAssistant messageBlockType = "assistant"
	blockUser      messageBlockType = "user"
)

// messageBlock is a run of consecutive goai messages that map to a single
// anthropic message. user-role and tool-role goai messages share a single
// `user` block — Anthropic strictly requires every tool_result for a given
// assistant turn to live in one user message immediately following that turn,
// so any user-role messages adjacent to those tool messages must merge in
// alongside them. Mirrors ai-sdk's groupIntoBlocks
// (references/ai-sdk/packages/anthropic/src/convert-to-anthropic-prompt.ts:1088).
type messageBlock struct {
	Type     messageBlockType
	Messages []message.Message
}

// groupIntoBlocks walks the prompt and groups consecutive messages by role,
// collapsing user+tool runs into a single `user` block and assistant runs
// into a single `assistant` block. The block walk in BuildRequestBody then
// emits exactly one anthropicMessage per block.
func groupIntoBlocks(messages []message.Message) []messageBlock {
	var blocks []messageBlock
	for _, msg := range messages {
		var want messageBlockType
		switch msg.Role {
		case message.RoleSystem:
			want = blockSystem
		case message.RoleAssistant:
			want = blockAssistant
		case message.RoleUser, message.RoleTool:
			// user + tool collapse into the same `user` block — see comment
			// on messageBlock.
			want = blockUser
		default:
			// Unknown role — drop. Matches the silent skip in the legacy
			// per-message switch (no default case there either).
			continue
		}

		if n := len(blocks); n > 0 && blocks[n-1].Type == want {
			blocks[n-1].Messages = append(blocks[n-1].Messages, msg)
			continue
		}
		blocks = append(blocks, messageBlock{Type: want, Messages: []message.Message{msg}})
	}
	return blocks
}
