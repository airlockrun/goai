// Package stream defines the streaming events from LLM responses.
// This mirrors the ai-sdk fullStream event types.
package stream

import (
	"encoding/json"

	"github.com/airlockrun/goai/message"
)

// EventType represents the type of streaming event.
type EventType string

const (
	EventStart            EventType = "start"
	EventTextStart        EventType = "text-start"
	EventTextDelta        EventType = "text-delta"
	EventTextEnd          EventType = "text-end"
	EventToolInputStart   EventType = "tool-input-start"
	EventToolInputDelta   EventType = "tool-input-delta"
	EventToolInputEnd     EventType = "tool-input-end"
	EventToolCall         EventType = "tool-call"
	EventToolResult       EventType = "tool-result"
	EventToolError        EventType = "tool-error"
	EventToolOutputDenied EventType = "tool-output-denied"
	EventReasoningStart   EventType = "reasoning-start"
	EventReasoningDelta   EventType = "reasoning-delta"
	EventReasoningEnd     EventType = "reasoning-end"
	EventStartStep        EventType = "start-step"
	EventFinishStep       EventType = "finish-step"
	EventFinish           EventType = "finish"
	EventError            EventType = "error"
	EventRawChunk         EventType = "raw"
	EventSource           EventType = "source"
)

// SourceType is the discriminator on SourceEvent. Mirrors ai-sdk's
// LanguageModelV4Source union (packages/provider/src/language-model/v4/
// language-model-v4-source.ts).
type SourceType string

const (
	// SourceTypeURL is a web source — used for url_citation annotations
	// emitted by web_search / google_search hosted tools.
	SourceTypeURL SourceType = "url"
	// SourceTypeDocument is a file/document source — used for
	// file_citation, container_file_citation, and file_path annotations.
	SourceTypeDocument SourceType = "document"
)

// Event represents a single streaming event.
type Event struct {
	Type EventType
	Data EventData
}

// EventData is the interface for all event data types.
type EventData interface {
	eventType() EventType
}

// WarningType mirrors ai-sdk's SharedV3Warning discriminant
// (references/ai-sdk/packages/provider/src/shared/v3/shared-v3-warning.ts).
type WarningType string

const (
	// WarningUnsupported reports a CallOption or provider-option the
	// model ignored (e.g. frequencyPenalty on Anthropic).
	WarningUnsupported WarningType = "unsupported"
	// WarningCompatibility reports an input that was silently converted
	// to make it compatible with the model (e.g. temperature clamped).
	WarningCompatibility WarningType = "compatibility"
	// WarningOther is a free-form informational warning.
	WarningOther WarningType = "other"
)

// Warning reports a non-fatal issue encountered while building or
// processing a provider request. Mirrors ai-sdk's SharedV3Warning.
//
// Feature is set for Unsupported/Compatibility; Message is set for Other.
// Details may be set for any type to provide extra context.
type Warning struct {
	Type    WarningType `json:"type"`
	Feature string      `json:"feature,omitempty"`
	Message string      `json:"message,omitempty"`
	Details string      `json:"details,omitempty"`
}

// UnsupportedWarning builds a Warning with type=unsupported.
func UnsupportedWarning(feature, details string) Warning {
	return Warning{Type: WarningUnsupported, Feature: feature, Details: details}
}

// CompatibilityWarning builds a Warning with type=compatibility.
func CompatibilityWarning(feature, details string) Warning {
	return Warning{Type: WarningCompatibility, Feature: feature, Details: details}
}

// OtherWarning builds a Warning with type=other.
func OtherWarning(message string) Warning {
	return Warning{Type: WarningOther, Message: message}
}

// StartEvent signals the start of streaming. Warnings surfaces
// non-fatal issues encountered while preparing the request (e.g.
// unsupported CallOptions, tools dropped by the converter).
type StartEvent struct {
	Warnings []Warning `json:"warnings,omitempty"`
}

func (StartEvent) eventType() EventType { return EventStart }

// TextStartEvent signals the start of text generation.
type TextStartEvent struct {
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (TextStartEvent) eventType() EventType { return EventTextStart }

// TextDeltaEvent contains a chunk of generated text.
type TextDeltaEvent struct {
	Text             string         `json:"text"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (TextDeltaEvent) eventType() EventType { return EventTextDelta }

// TextEndEvent signals the end of text generation.
type TextEndEvent struct {
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (TextEndEvent) eventType() EventType { return EventTextEnd }

// ToolInputStartEvent signals the start of tool input streaming.
type ToolInputStartEvent struct {
	ProviderExecuted bool   `json:"providerExecuted,omitempty"`
	ID               string `json:"id"`
	ToolName         string `json:"toolName"`
}

func (ToolInputStartEvent) eventType() EventType { return EventToolInputStart }

// ToolInputDeltaEvent contains a chunk of tool input.
type ToolInputDeltaEvent struct {
	ID    string `json:"id"`
	Delta string `json:"delta"`
}

func (ToolInputDeltaEvent) eventType() EventType { return EventToolInputDelta }

// ToolInputEndEvent signals the end of tool input streaming.
type ToolInputEndEvent struct {
	ID string `json:"id"`
}

func (ToolInputEndEvent) eventType() EventType { return EventToolInputEnd }

// ToolCallEvent signals a complete tool call ready for execution.
type ToolCallEvent struct {
	ProviderExecuted bool            `json:"providerExecuted,omitempty"`
	ToolCallID       string          `json:"toolCallId"`
	ToolName         string          `json:"toolName"`
	Input            json.RawMessage `json:"input"`
	ProviderMetadata map[string]any  `json:"providerMetadata,omitempty"`
}

func (ToolCallEvent) eventType() EventType { return EventToolCall }

// ToolResultEvent contains a successful tool execution result. Output is a
// success variant of the discriminated union (text | json | content).
//
// Title and Metadata carry the tool's presentation hints (the tool's
// short Result.Title and its structured Result.Metadata). Consumers that
// render a curated activity log — e.g. airlock's build log — use these to
// summarize a call without dumping the full model-facing Output.
type ToolResultEvent struct {
	ProviderExecuted bool                     `json:"providerExecuted,omitempty"`
	ProviderMetadata map[string]any           `json:"providerMetadata,omitempty"`
	ToolCallID       string                   `json:"toolCallId"`
	ToolName         string                   `json:"toolName"`
	Input            json.RawMessage          `json:"input,omitempty"`
	Output           message.ToolResultOutput `json:"output"`
	Title            string                   `json:"title,omitempty"`
	Metadata         map[string]any           `json:"metadata,omitempty"`
}

func (ToolResultEvent) eventType() EventType { return EventToolResult }

func (e ToolResultEvent) MarshalJSON() ([]byte, error) {
	type alias ToolResultEvent
	out, err := message.MarshalOutput(e.Output)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		alias
		Output json.RawMessage `json:"output"`
	}{alias(e), out})
}
func (e *ToolResultEvent) UnmarshalJSON(b []byte) error {
	type alias ToolResultEvent
	var a struct {
		alias
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*e = ToolResultEvent(a.alias)
	if len(a.Output) == 0 || string(a.Output) == "null" {
		return nil
	}
	output, err := message.UnmarshalOutput(a.Output)
	if err != nil {
		return err
	}
	e.Output = output
	return nil
}

// ToolErrorEvent signals a failed tool execution. Output carries the
// error variant (error-text | error-json).
type ToolErrorEvent struct {
	ProviderExecuted bool                     `json:"providerExecuted,omitempty"`
	ProviderMetadata map[string]any           `json:"providerMetadata,omitempty"`
	ToolCallID       string                   `json:"toolCallId"`
	ToolName         string                   `json:"toolName"`
	Input            json.RawMessage          `json:"input,omitempty"`
	Output           message.ToolResultOutput `json:"output"`
}

func (ToolErrorEvent) eventType() EventType { return EventToolError }

func (e ToolErrorEvent) MarshalJSON() ([]byte, error) {
	return json.Marshal(ToolResultEvent{ToolCallID: e.ToolCallID, ToolName: e.ToolName, Input: e.Input, Output: e.Output, ProviderExecuted: e.ProviderExecuted, ProviderMetadata: e.ProviderMetadata})
}
func (e *ToolErrorEvent) UnmarshalJSON(b []byte) error {
	var result ToolResultEvent
	if err := json.Unmarshal(b, &result); err != nil {
		return err
	}
	*e = ToolErrorEvent{ToolCallID: result.ToolCallID, ToolName: result.ToolName, Input: result.Input, Output: result.Output, ProviderExecuted: result.ProviderExecuted, ProviderMetadata: result.ProviderMetadata}
	return nil
}

// ErrorText returns the error message text for this event.
func (e ToolErrorEvent) ErrorText() string { return message.ToolOutputText(e.Output) }

// ToolOutputDeniedEvent signals a tool call the user/policy refused to run.
type ToolOutputDeniedEvent struct {
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Input      json.RawMessage `json:"input,omitempty"`
	Reason     string          `json:"reason,omitempty"`
}

func (ToolOutputDeniedEvent) eventType() EventType { return EventToolOutputDenied }

// ToolOutcomeEvent wraps a tool output into the stream event whose kind
// matches its discriminated outcome: success → tool-result, error →
// tool-error, execution-denied → tool-output-denied. Shared by every
// producer so consumers that branch on event type get a consistent signal
// while the carried Output keeps the full union.
// title and metadata ride only on the success outcome — the discriminated
// error/denied variants carry no tool-supplied metadata.
func ToolOutcomeEvent(toolCallID, toolName string, input json.RawMessage, out message.ToolResultOutput, title string, metadata map[string]any) Event {
	switch message.ToolOutcome(out) {
	case "error":
		return Event{Type: EventToolError, Data: ToolErrorEvent{
			ToolCallID: toolCallID, ToolName: toolName, Input: input, Output: out,
		}}
	case "denied":
		reason := ""
		if d, ok := out.(message.ExecutionDeniedOutput); ok {
			reason = d.Reason
		}
		return Event{Type: EventToolOutputDenied, Data: ToolOutputDeniedEvent{
			ToolCallID: toolCallID, ToolName: toolName, Input: input, Reason: reason,
		}}
	default:
		return Event{Type: EventToolResult, Data: ToolResultEvent{
			ToolCallID: toolCallID, ToolName: toolName, Input: input, Output: out,
			Title: title, Metadata: metadata,
		}}
	}
}

// ReasoningStartEvent signals the start of reasoning/thinking.
type ReasoningStartEvent struct {
	ID               string         `json:"id"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (ReasoningStartEvent) eventType() EventType { return EventReasoningStart }

// ReasoningDeltaEvent contains a chunk of reasoning text.
type ReasoningDeltaEvent struct {
	ID               string         `json:"id"`
	Text             string         `json:"text"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (ReasoningDeltaEvent) eventType() EventType { return EventReasoningDelta }

// ReasoningEndEvent signals the end of reasoning.
type ReasoningEndEvent struct {
	ID               string         `json:"id"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (ReasoningEndEvent) eventType() EventType { return EventReasoningEnd }

// StartStepEvent signals the start of a processing step.
type StartStepEvent struct{}

func (StartStepEvent) eventType() EventType { return EventStartStep }

// FinishStepEvent signals the end of a processing step.
type FinishStepEvent struct {
	FinishReason     FinishReason   `json:"finishReason"`
	Usage            Usage          `json:"usage"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (FinishStepEvent) eventType() EventType { return EventFinishStep }

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
)

// Usage contains token usage information. Mirrors ai-sdk's
// LanguageModelV4Usage (packages/provider/src/language-model/v4/language-model-v4-usage.ts).
// All integer fields are pointers so "unreported by provider" (nil) is
// distinguishable from "reported as zero" (pointer to 0).
type Usage struct {
	InputTokens  InputTokens    `json:"inputTokens"`
	OutputTokens OutputTokens   `json:"outputTokens"`
	Raw          map[string]any `json:"raw,omitempty"`
}

// Normalized clamps reported token counts to zero without changing Raw or the
// provider's pointers. Unreported counts remain nil.
func (u Usage) Normalized() Usage {
	u.InputTokens.Total = addIntPtrs(u.InputTokens.Total, nil)
	u.InputTokens.NoCache = addIntPtrs(u.InputTokens.NoCache, nil)
	u.InputTokens.CacheRead = addIntPtrs(u.InputTokens.CacheRead, nil)
	u.InputTokens.CacheWrite = addIntPtrs(u.InputTokens.CacheWrite, nil)
	u.OutputTokens.Total = addIntPtrs(u.OutputTokens.Total, nil)
	u.OutputTokens.Text = addIntPtrs(u.OutputTokens.Text, nil)
	u.OutputTokens.Reasoning = addIntPtrs(u.OutputTokens.Reasoning, nil)
	return u
}

// InputTokens holds the prompt-side token breakdown.
type InputTokens struct {
	// Total is the total number of input (prompt) tokens.
	Total *int `json:"total,omitempty"`
	// NoCache is the number of non-cached input tokens.
	NoCache *int `json:"noCache,omitempty"`
	// CacheRead is the number of cached input tokens read.
	CacheRead *int `json:"cacheRead,omitempty"`
	// CacheWrite is the number of cached input tokens written.
	CacheWrite *int `json:"cacheWrite,omitempty"`
}

// OutputTokens holds the completion-side token breakdown.
type OutputTokens struct {
	// Total is the total number of output (completion) tokens.
	Total *int `json:"total,omitempty"`
	// Text is the number of text-output tokens.
	Text *int `json:"text,omitempty"`
	// Reasoning is the number of reasoning/thinking tokens.
	Reasoning *int `json:"reasoning,omitempty"`
}

// IntPtr returns a pointer to an int. Convenience helper for constructing
// Usage values from raw integers, since all Usage token fields are *int.
func IntPtr(v int) *int { return &v }

// UsageFrom builds a Usage from a prompt/completion pair. Convenience
// helper for providers that only surface the totals; more detailed
// breakdowns should construct Usage explicitly.
func UsageFrom(prompt, completion int) Usage {
	return Usage{
		InputTokens:  InputTokens{Total: IntPtr(prompt)},
		OutputTokens: OutputTokens{Total: IntPtr(completion)},
	}
}

// InputTotal returns the total input tokens, or 0 if unreported.
func (u Usage) InputTotal() int {
	if u.InputTokens.Total == nil {
		return 0
	}
	return *u.InputTokens.Total
}

// OutputTotal returns the total output tokens, or 0 if unreported.
func (u Usage) OutputTotal() int {
	if u.OutputTokens.Total == nil {
		return 0
	}
	return *u.OutputTokens.Total
}

// GrandTotal returns InputTotal + OutputTotal for callers that want a
// single aggregate number (replaces the old TotalTokens field).
func (u Usage) GrandTotal() int {
	total := addIntPtrs(u.InputTokens.Total, u.OutputTokens.Total)
	if total == nil {
		return 0
	}
	return *total
}

// Add accumulates another Usage into the receiver. Nil fields on either
// side are treated as zero for the purposes of the sum; the result
// field is nil only when both sides had nil. This mirrors the common
// multi-step aggregation pattern in goai.GenerateText / StreamText.
func (u *Usage) Add(other Usage) {
	// Raw belongs to an individual provider response, never an aggregate.
	u.Raw = nil
	u.InputTokens.Total = addIntPtrs(u.InputTokens.Total, other.InputTokens.Total)
	u.InputTokens.NoCache = addIntPtrs(u.InputTokens.NoCache, other.InputTokens.NoCache)
	u.InputTokens.CacheRead = addIntPtrs(u.InputTokens.CacheRead, other.InputTokens.CacheRead)
	u.InputTokens.CacheWrite = addIntPtrs(u.InputTokens.CacheWrite, other.InputTokens.CacheWrite)
	u.OutputTokens.Total = addIntPtrs(u.OutputTokens.Total, other.OutputTokens.Total)
	u.OutputTokens.Text = addIntPtrs(u.OutputTokens.Text, other.OutputTokens.Text)
	u.OutputTokens.Reasoning = addIntPtrs(u.OutputTokens.Reasoning, other.OutputTokens.Reasoning)
}

func addIntPtrs(a, b *int) *int {
	if a == nil && b == nil {
		return nil
	}
	av := 0
	if a != nil {
		av = max(0, *a)
	}
	bv := 0
	if b != nil {
		bv = max(0, *b)
	}
	maxInt := int(^uint(0) >> 1)
	sum := maxInt
	if bv <= maxInt-av {
		sum = av + bv
	}
	return &sum
}

// FinishEvent signals the end of the stream.
type FinishEvent struct {
	FinishReason     FinishReason   `json:"finishReason"`
	Usage            Usage          `json:"usage"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (FinishEvent) eventType() EventType { return EventFinish }

// ErrorEvent signals an error during streaming.
type ErrorEvent struct {
	Error error `json:"error"`
}

func (ErrorEvent) eventType() EventType { return EventError }

// SourceEvent reports a source the model used to generate the response —
// emitted by providers when a hosted tool (web_search, google_search,
// file_search, ...) cites a URL or document. Mirrors ai-sdk's
// LanguageModelV4Source content part.
//
// SourceType discriminates which fields are meaningful:
//   - SourceTypeURL: ID, URL, Title.
//   - SourceTypeDocument: ID, MediaType, Title, Filename.
//
// ProviderMetadata carries provider-specific extras (e.g. OpenAI's
// file_id and container_id on file_citation annotations).
type SourceEvent struct {
	SourceType       SourceType     `json:"sourceType"`
	ID               string         `json:"id"`
	URL              string         `json:"url,omitempty"`
	Title            string         `json:"title,omitempty"`
	MediaType        string         `json:"mediaType,omitempty"`
	Filename         string         `json:"filename,omitempty"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

func (SourceEvent) eventType() EventType { return EventSource }

// RawChunkEvent carries an unparsed payload from the upstream provider.
// Emitted only when CallOptions.IncludeRawChunks is true. RawValue is
// typically the SSE "data: …" string with the prefix already trimmed;
// providers may emit []byte or a parsed object when that's more useful.
// Mirrors ai-sdk's v4 raw stream-part (LanguageModelV4StreamPart).
type RawChunkEvent struct {
	RawValue any `json:"rawValue"`
}

func (RawChunkEvent) eventType() EventType { return EventRawChunk }
