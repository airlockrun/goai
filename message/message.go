// Package message defines the core message types used in AI conversations.
// This mirrors the ai-sdk ModelMessage types.
package message

import (
	"encoding/json"
	"fmt"
	"strings"
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
	ContentTypeText                 ContentType = "text"
	ContentTypeImage                ContentType = "image"
	ContentTypeFile                 ContentType = "file"
	ContentTypeToolCall             ContentType = "tool-call"
	ContentTypeToolResult           ContentType = "tool-result"
	ContentTypeReasoning            ContentType = "reasoning"
	ContentTypeToolApprovalRequest  ContentType = "tool-approval-request"
	ContentTypeToolApprovalResponse ContentType = "tool-approval-response"
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

// ToolResultPart represents the result of a tool invocation. Output is a
// discriminated union mirroring ai-sdk's ToolResultOutput (references/ai-sdk/
// packages/provider-utils/src/types/content-part.ts): the concrete variant
// is the outcome (success text/json/content vs error vs execution-denied).
type ToolResultPart struct {
	ToolCallID       string           `json:"toolCallId"`
	ToolName         string           `json:"toolName"`
	Output           ToolResultOutput `json:"output"`
	ProviderExecuted bool             `json:"providerExecuted,omitempty"`
	ProviderOptions  map[string]any   `json:"providerOptions,omitempty"`
}

func (ToolResultPart) partType() ContentType { return ContentTypeToolResult }

// MarshalJSON serializes the part, injecting the discriminated "type" into
// the nested output object.
func (p ToolResultPart) MarshalJSON() ([]byte, error) {
	out, err := marshalToolOutput(p.Output)
	if err != nil {
		return nil, err
	}
	type alias struct {
		ToolCallID       string          `json:"toolCallId"`
		ToolName         string          `json:"toolName"`
		Output           json.RawMessage `json:"output"`
		ProviderExecuted bool            `json:"providerExecuted,omitempty"`
		ProviderOptions  map[string]any  `json:"providerOptions,omitempty"`
	}
	return json.Marshal(alias{p.ToolCallID, p.ToolName, out, p.ProviderExecuted, p.ProviderOptions})
}

// UnmarshalJSON deserializes the part. The output is the discriminated
// ToolResultOutput union; a missing or malformed output is a hard error
// (fail loud — the data migration converts all history to this shape, so
// there is no legacy {result,isError} path to tolerate).
func (p *ToolResultPart) UnmarshalJSON(data []byte) error {
	var a struct {
		ToolCallID       string          `json:"toolCallId"`
		ToolName         string          `json:"toolName"`
		Output           json.RawMessage `json:"output"`
		ProviderExecuted bool            `json:"providerExecuted,omitempty"`
		ProviderOptions  map[string]any  `json:"providerOptions,omitempty"`
	}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	p.ToolCallID = a.ToolCallID
	p.ToolName = a.ToolName
	p.ProviderExecuted = a.ProviderExecuted
	p.ProviderOptions = a.ProviderOptions
	if len(a.Output) == 0 || string(a.Output) == "null" {
		return fmt.Errorf("tool-result part %q: missing output", a.ToolCallID)
	}
	o, err := unmarshalToolOutput(a.Output)
	if err != nil {
		return err
	}
	p.Output = o
	return nil
}

// ToolResultOutput is the discriminated output of a tool result, mirroring
// ai-sdk's ToolResultOutput union. The Go concrete type is the discriminant;
// JSON carries it as the nested "type" field.
type ToolResultOutput interface {
	outputType() string
}

// TextOutput is a successful text result (type:"text").
type TextOutput struct {
	Value           string         `json:"value"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (TextOutput) outputType() string { return "text" }

// JSONOutput is a successful structured result (type:"json").
type JSONOutput struct {
	Value           any            `json:"value"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (JSONOutput) outputType() string { return "json" }

// ErrorTextOutput is a failed tool execution with a text message
// (type:"error-text"). Drives the provider wire is_error flag.
type ErrorTextOutput struct {
	Value           string         `json:"value"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ErrorTextOutput) outputType() string { return "error-text" }

// ErrorJSONOutput is a failed tool execution with a structured payload
// (type:"error-json"). Drives the provider wire is_error flag.
type ErrorJSONOutput struct {
	Value           any            `json:"value"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ErrorJSONOutput) outputType() string { return "error-json" }

// ExecutionDeniedOutput is a tool call the user/policy refused to run
// (type:"execution-denied"). Distinct from an error so the agent re-reasons
// and the UI can show it differently. Reason is optional.
type ExecutionDeniedOutput struct {
	Reason          string         `json:"reason,omitempty"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ExecutionDeniedOutput) outputType() string { return "execution-denied" }

// ContentOutput is a multipart successful result (type:"content"): text and
// file/image items. Mirrors ai-sdk's content variant inner union.
type ContentOutput struct {
	Value []ToolContentItem `json:"value"`
}

func (ContentOutput) outputType() string { return "content" }

// ToolContentItem is one item of a ContentOutput. Type discriminates which
// fields are meaningful, mirroring ai-sdk's inner union (text | file-data |
// file-url | file-id | file-reference | image-data | image-url |
// image-file-id | image-file-reference | custom).
type ToolContentItem struct {
	Type              string            `json:"type"`
	Text              string            `json:"text,omitempty"`
	Data              string            `json:"data,omitempty"`      // file-data / image-data: base64
	MediaType         string            `json:"mediaType,omitempty"` // file-data / file-url
	Filename          string            `json:"filename,omitempty"`  // file-data
	URL               string            `json:"url,omitempty"`       // file-url / image-url
	FileID            any               `json:"fileId,omitempty"`    // string | map[string]string
	ProviderReference map[string]string `json:"providerReference,omitempty"`
	ProviderOptions   map[string]any    `json:"providerOptions,omitempty"`
}

// marshalToolOutput serializes a ToolResultOutput with its "type"
// discriminant injected as the first field. A nil Output marshals as an
// empty text output so a part always round-trips.
func marshalToolOutput(o ToolResultOutput) (json.RawMessage, error) {
	if o == nil {
		o = TextOutput{}
	}
	inner, err := json.Marshal(o)
	if err != nil {
		return nil, err
	}
	return injectType(o.outputType(), inner), nil
}

// MarshalOutput serializes a ToolResultOutput as a self-describing object
// (with the "type" discriminant). Exported for stream-event and wire
// serializers that embed an Output outside a ToolResultPart.
func MarshalOutput(o ToolResultOutput) (json.RawMessage, error) { return marshalToolOutput(o) }

// UnmarshalOutput decodes a ToolResultOutput produced by MarshalOutput.
func UnmarshalOutput(raw json.RawMessage) (ToolResultOutput, error) { return unmarshalToolOutput(raw) }

// unmarshalToolOutput decodes a ToolResultOutput keyed by its "type" field.
func unmarshalToolOutput(raw json.RawMessage) (ToolResultOutput, error) {
	var peek struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &peek); err != nil {
		return nil, fmt.Errorf("invalid tool output: %w", err)
	}
	switch peek.Type {
	case "text":
		var o TextOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	case "json":
		var o JSONOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	case "error-text":
		var o ErrorTextOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	case "error-json":
		var o ErrorJSONOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	case "execution-denied":
		var o ExecutionDeniedOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	case "content":
		var o ContentOutput
		if err := json.Unmarshal(raw, &o); err != nil {
			return nil, err
		}
		return o, nil
	default:
		return nil, fmt.Errorf("unknown tool output type: %q", peek.Type)
	}
}

// ToolOutputText renders any ToolResultOutput to a single string — the
// model-facing text for that result. JSON variants are compact-encoded;
// execution-denied falls back to a default sentence when Reason is empty;
// content concatenates its text items.
func ToolOutputText(o ToolResultOutput) string {
	switch v := o.(type) {
	case TextOutput:
		return v.Value
	case ErrorTextOutput:
		return v.Value
	case JSONOutput:
		b, _ := json.Marshal(v.Value)
		return string(b)
	case ErrorJSONOutput:
		b, _ := json.Marshal(v.Value)
		return string(b)
	case ExecutionDeniedOutput:
		if v.Reason != "" {
			return v.Reason
		}
		return "Tool call execution denied."
	case ContentOutput:
		var parts []string
		for _, it := range v.Value {
			if it.Type == "text" {
				parts = append(parts, it.Text)
			}
		}
		return strings.Join(parts, "")
	default:
		return ""
	}
}

// ToolOutputIsError reports whether the output is an error variant — the
// single source of truth for a provider's wire is_error flag.
func ToolOutputIsError(o ToolResultOutput) bool {
	switch o.(type) {
	case ErrorTextOutput, ErrorJSONOutput:
		return true
	default:
		return false
	}
}

// ToolOutputWire renders a ToolResultOutput to the single string most
// providers put in their tool message (no is_error field): text/error-text
// → value; execution-denied → reason or default; json/error-json/content →
// compact JSON. Mirrors ai-sdk's convert-to-openai-chat-messages behavior.
func ToolOutputWire(o ToolResultOutput) string {
	switch v := o.(type) {
	case TextOutput:
		return v.Value
	case ErrorTextOutput:
		return v.Value
	case ExecutionDeniedOutput:
		if v.Reason != "" {
			return v.Reason
		}
		return "Tool call execution denied."
	case JSONOutput:
		b, _ := json.Marshal(v.Value)
		return string(b)
	case ErrorJSONOutput:
		b, _ := json.Marshal(v.Value)
		return string(b)
	case ContentOutput:
		b, _ := json.Marshal(v.Value)
		return string(b)
	default:
		return ""
	}
}

// ToolOutcome classifies a ToolResultOutput as "success", "error" or
// "denied" — the persisted, heuristic-free tool status.
func ToolOutcome(o ToolResultOutput) string {
	switch o.(type) {
	case ErrorTextOutput, ErrorJSONOutput:
		return "error"
	case ExecutionDeniedOutput:
		return "denied"
	default:
		return "success"
	}
}

// ReasoningPart represents reasoning/thinking content from the model.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ReasoningPart struct {
	Text            string         `json:"text"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ReasoningPart) partType() ContentType { return ContentTypeReasoning }

// ToolApprovalRequestPart represents a request for tool execution approval.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ToolApprovalRequestPart struct {
	ApprovalID      string         `json:"approvalId"`
	ToolCallID      string         `json:"toolCallId"`
	ToolName        string         `json:"toolName"`
	Input           any            `json:"input"`
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

func (ToolApprovalRequestPart) partType() ContentType { return ContentTypeToolApprovalRequest }

// ToolApprovalResponsePart represents a response to tool execution approval.
// Source: ai-sdk/packages/provider-utils/src/message.ts
type ToolApprovalResponsePart struct {
	ApprovalID       string         `json:"approvalId"`
	Approved         bool           `json:"approved"`
	Reason           string         `json:"reason,omitempty"`
	ProviderExecuted bool           `json:"providerExecuted,omitempty"`
	ProviderOptions  map[string]any `json:"providerOptions,omitempty"`
}

func (ToolApprovalResponsePart) partType() ContentType { return ContentTypeToolApprovalResponse }

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
		case ContentTypeReasoning:
			var p ReasoningPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeToolApprovalRequest:
			var p ToolApprovalRequestPart
			if err := json.Unmarshal(raw, &p); err != nil {
				return err
			}
			part = p
		case ContentTypeToolApprovalResponse:
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
	inner, err := json.Marshal(e.Part)
	if err != nil {
		return nil, err
	}
	return injectType(e.Type, inner), nil
}

// injectType returns inner (a JSON object) with a "type" field spliced in as
// the first key. Shared by partEnvelope and tool-output marshaling so the
// discriminated "type" is always the leading field.
func injectType(typeName string, inner []byte) []byte {
	typeBytes, _ := json.Marshal(typeName)
	if len(inner) < 2 {
		out := append([]byte(`{"type":`), typeBytes...)
		return append(out, '}')
	}
	result := make([]byte, 0, len(inner)+len(typeName)+10)
	result = append(result, '{')
	result = append(result, '"', 't', 'y', 'p', 'e', '"', ':')
	result = append(result, typeBytes...)
	if len(inner) > 2 { // has fields beyond {}
		result = append(result, ',')
		result = append(result, inner[1:]...) // skip opening {
	} else {
		result = append(result, '}')
	}
	return result
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

// NewToolMessage creates a tool result message wrapping the given
// discriminated output (success / error / denied).
func NewToolMessage(toolCallID, toolName string, output ToolResultOutput) Message {
	return Message{
		Role: RoleTool,
		Content: Content{
			Parts: []Part{
				ToolResultPart{
					ToolCallID: toolCallID,
					ToolName:   toolName,
					Output:     output,
				},
			},
		},
	}
}

// NewToolResultText creates a tool result message with a text success
// output.
func NewToolResultText(toolCallID, toolName, value string) Message {
	return NewToolMessage(toolCallID, toolName, TextOutput{Value: value})
}

// NewToolResultJSON creates a tool result message with a structured
// success output.
func NewToolResultJSON(toolCallID, toolName string, value any) Message {
	return NewToolMessage(toolCallID, toolName, JSONOutput{Value: value})
}

// NewToolResultDenied creates a tool result message marking the call as
// execution-denied. reason may be empty.
func NewToolResultDenied(toolCallID, toolName, reason string) Message {
	return NewToolMessage(toolCallID, toolName, ExecutionDeniedOutput{Reason: reason})
}
