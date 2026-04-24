// Package message defines the core message types used in AI conversations.
// This mirrors the ai-sdk ModelMessage types.
package message

import (
	"encoding/json"
	"fmt"
)

// Role represents the role of a message sender.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentType represents the type of content in a message part.
type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeImage      ContentType = "image"
	ContentTypeFile       ContentType = "file"
	ContentTypeToolCall   ContentType = "tool-call"
	ContentTypeToolResult ContentType = "tool-result"
)

// Part represents a single part of message content.
type Part interface {
	partType() ContentType
}

// TextPart represents text content.
type TextPart struct {
	Text            string         `json:"text"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (TextPart) partType() ContentType { return ContentTypeText }

// ImagePart represents image content.
type ImagePart struct {
	Image           string         `json:"image"`              // base64 or URL
	MimeType        string         `json:"mimeType,omitempty"` // e.g., "image/png"
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ImagePart) partType() ContentType { return ContentTypeImage }

// FilePart represents file content. Exactly one of Data or URL should
// be set: Data carries base64-encoded bytes; URL points to a remote file
// the provider should fetch (matches ai-sdk's file-url content part and
// the Anthropic/Gemini url-source variants).
type FilePart struct {
	Data            string         `json:"data,omitempty"`     // base64 encoded (when inline)
	URL             string         `json:"url,omitempty"`      // remote URL (when hosted)
	MimeType        string         `json:"mimeType"`           // e.g., "application/pdf"
	Filename        string         `json:"filename,omitempty"` // optional filename
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (FilePart) partType() ContentType { return ContentTypeFile }

// ToolCallPart represents a tool invocation by the assistant.
type ToolCallPart struct {
	ID              string          `json:"toolCallId"`
	Name            string          `json:"toolName"`
	Input           json.RawMessage `json:"args"`
	ProviderOptions map[string]any  `json:"providerOptions,omitempty"`
}

func (ToolCallPart) partType() ContentType { return ContentTypeToolCall }

// ToolResultPart represents the result of a tool invocation.
type ToolResultPart struct {
	ToolCallID      string         `json:"toolCallId"`
	ToolName        string         `json:"toolName"`
	Result          any            `json:"result"`
	IsError         bool           `json:"isError,omitempty"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ToolResultPart) partType() ContentType { return ContentTypeToolResult }

// ReasoningPart represents reasoning/thinking content from the model.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ReasoningPart struct {
	Text            string         `json:"text"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ReasoningPart) partType() ContentType { return "reasoning" }

// ToolApprovalRequestPart represents a request for tool execution approval.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ToolApprovalRequestPart struct {
	ApprovalID      string         `json:"approvalId"`
	ToolCallID      string         `json:"toolCallId"`
	ToolName        string         `json:"toolName"`
	Input           any            `json:"input"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ToolApprovalRequestPart) partType() ContentType { return "tool-approval-request" }

// ToolApprovalResponsePart represents a response to tool execution approval.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ToolApprovalResponsePart struct {
	ApprovalID       string         `json:"approvalId"`
	Approved         bool           `json:"approved"`
	Reason           string         `json:"reason,omitempty"`
	ProviderExecuted bool           `json:"providerExecuted,omitempty"`
	ProviderOptions  map[string]any `json:"providerOptions,omitempty"`
}

func (ToolApprovalResponsePart) partType() ContentType { return "tool-approval-response" }

// Content represents message content - either a string or parts.
type Content struct {
	Text  string // Simple text content
	Parts []Part // Multi-part content
}

// IsMultiPart returns true if content has multiple parts.
func (c Content) IsMultiPart() bool {
	return len(c.Parts) > 0
}

// MarshalJSON serializes Content. Text-only content serializes as a plain string.
// Multi-part content serializes as an array of typed objects.
func (c Content) MarshalJSON() ([]byte, error) {
	if !c.IsMultiPart() {
		return json.Marshal(c.Text)
	}
	envelopes := make([]partEnvelope, len(c.Parts))
	for i, p := range c.Parts {
		envelopes[i] = partEnvelope{Type: string(p.partType()), Part: p}
	}
	return json.Marshal(envelopes)
}

// UnmarshalJSON deserializes Content from either a plain string or a typed array.
func (c *Content) UnmarshalJSON(data []byte) error {
	// Try string first (text-only content)
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		c.Text = text
		c.Parts = nil
		return nil
	}

	// Must be an array of part envelopes
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return fmt.Errorf("content must be a string or array: %w", err)
	}

	c.Parts = make([]Part, 0, len(raws))
	for _, raw := range raws {
		// Peek at the type field
		var peek struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &peek); err != nil {
			return fmt.Errorf("invalid part: %w", err)
		}

		var part Part
		switch ContentType(peek.Type) {
		case ContentTypeText:
			var p TextPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeImage:
			var p ImagePart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeFile:
			var p FilePart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeToolCall:
			var p ToolCallPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeToolResult:
			var p ToolResultPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case "reasoning":
			var p ReasoningPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case "tool-approval-request":
			var p ToolApprovalRequestPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case "tool-approval-response":
			var p ToolApprovalResponsePart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		default:
			return fmt.Errorf("unknown part type: %q", peek.Type)
		}
		c.Parts = append(c.Parts, part)
	}
	c.Text = ""
	return nil
}

// partEnvelope wraps a Part with a type discriminator for JSON serialization.
type partEnvelope struct {
	Type string `json:"type"`
	Part
}

// MarshalJSON flattens the envelope so "type" is a sibling of the part fields.
func (e partEnvelope) MarshalJSON() ([]byte, error) {
	// Marshal the inner part
	inner, err := json.Marshal(e.Part)
	if err != nil {
		return nil, err
	}
	// Inject "type" field
	if len(inner) < 2 {
		return json.Marshal(map[string]string{"type": e.Type})
	}
	// inner is like {"text":"..."} — inject type as first field
	result := make([]byte, 0, len(inner)+len(e.Type)+10)
	result = append(result, '{')
	result = append(result, '"', 't', 'y', 'p', 'e', '"', ':')
	typeBytes, _ := json.Marshal(e.Type)
	result = append(result, typeBytes...)
	if len(inner) > 2 { // has fields beyond {}
		result = append(result, ',')
		result = append(result, inner[1:]...) // skip opening {
	} else {
		result = append(result, '}')
	}
	return result, nil
}

// Message represents a single message in a conversation.
type Message struct {
	Role            Role           `json:"role"`
	Content         Content        `json:"content"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

// NewSystemMessage creates a system message.
func NewSystemMessage(text string) Message {
	return Message{
		Role:    RoleSystem,
		Content: Content{Text: text},
	}
}

// NewUserMessage creates a user message with text.
func NewUserMessage(text string) Message {
	return Message{
		Role:    RoleUser,
		Content: Content{Text: text},
	}
}

// NewUserMessageWithParts creates a user message with multiple parts.
func NewUserMessageWithParts(parts ...Part) Message {
	return Message{
		Role:    RoleUser,
		Content: Content{Parts: parts},
	}
}

// NewAssistantMessage creates an assistant message.
func NewAssistantMessage(text string) Message {
	return Message{
		Role:    RoleAssistant,
		Content: Content{Text: text},
	}
}

// NewAssistantMessageWithParts creates an assistant message with parts (e.g., tool calls).
func NewAssistantMessageWithParts(parts ...Part) Message {
	return Message{
		Role:    RoleAssistant,
		Content: Content{Parts: parts},
	}
}

// NewToolMessage creates a tool result message.
func NewToolMessage(toolCallID, toolName string, result any, isError bool) Message {
	return Message{
		Role: RoleTool,
		Content: Content{
			Parts: []Part{
				ToolResultPart{
					ToolCallID: toolCallID,
					ToolName:   toolName,
					Result:     result,
					IsError:    isError,
				},
			},
		},
	}
}
