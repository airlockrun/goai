// Package stream defines the streaming events from LLM responses.
// This mirrors the ai-sdk fullStream event types.
package stream

import "encoding/json"

// EventType represents the type of streaming event.
type EventType string

const (
	EventStart          EventType = "start"
	EventTextStart      EventType = "text-start"
	EventTextDelta      EventType = "text-delta"
	EventTextEnd        EventType = "text-end"
	EventToolInputStart EventType = "tool-input-start"
	EventToolInputDelta EventType = "tool-input-delta"
	EventToolInputEnd   EventType = "tool-input-end"
	EventToolCall       EventType = "tool-call"
	EventToolResult     EventType = "tool-result"
	EventToolError      EventType = "tool-error"
	EventReasoningStart EventType = "reasoning-start"
	EventReasoningDelta EventType = "reasoning-delta"
	EventReasoningEnd   EventType = "reasoning-end"
	EventStartStep      EventType = "start-step"
	EventFinishStep     EventType = "finish-step"
	EventFinish         EventType = "finish"
	EventError          EventType = "error"
	EventRawChunk       EventType = "raw"
	EventSource         EventType = "source"
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
	ID       string `json:"id"`
	ToolName string `json:"toolName"`
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
	ToolCallID       string          `json:"toolCallId"`
	ToolName         string          `json:"toolName"`
	Input            json.RawMessage `json:"input"`
	ProviderMetadata map[string]any  `json:"providerMetadata,omitempty"`
}

func (ToolCallEvent) eventType() EventType { return EventToolCall }

// ToolResultEvent contains the result of a tool execution.
type ToolResultEvent struct {
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     ToolOutput      `json:"output"`
}

func (ToolResultEvent) eventType() EventType { return EventToolResult }

// ToolOutput represents the output from a tool execution.
type ToolOutput struct {
	Output      string         `json:"output"`
	Title       string         `json:"title,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Attachments []Attachment   `json:"attachments,omitempty"`
}

// Attachment represents a file attachment.
type Attachment struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
	Filename string `json:"filename,omitempty"`
}

// ToolErrorEvent signals an error during tool execution.
type ToolErrorEvent struct {
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Input      json.RawMessage `json:"input,omitempty"`
	Error      error           `json:"error"`
}

func (ToolErrorEvent) eventType() EventType { return EventToolError }

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
	return u.InputTotal() + u.OutputTotal()
}

// Add accumulates another Usage into the receiver. Nil fields on either
// side are treated as zero for the purposes of the sum; the result
// field is nil only when both sides had nil. This mirrors the common
// multi-step aggregation pattern in goai.GenerateText / StreamText.
func (u *Usage) Add(other Usage) {
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
		av = *a
	}
	bv := 0
	if b != nil {
		bv = *b
	}
	sum := av + bv
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
