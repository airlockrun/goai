package stream

import (
	"context"
	"encoding/json"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/tool"
)

// Result represents the result of a streaming text generation.
type Result struct {
	// FullStream is a channel that receives all streaming events.
	FullStream <-chan Event

	// Text returns the final complete text when streaming is done.
	// This blocks until the stream is complete.
	Text func() string

	// ToolCalls returns all tool calls made during generation.
	// This blocks until the stream is complete.
	ToolCalls func() []ToolCall

	// ToolResults returns all tool results.
	// This blocks until the stream is complete.
	ToolResults func() []ToolResultEvent

	// FinishReason returns why the generation stopped.
	// This blocks until the stream is complete.
	FinishReason func() FinishReason

	// Usage returns token usage statistics.
	// This blocks until the stream is complete.
	Usage func() Usage

	// Output returns the parsed output when an Output strategy was provided
	// on Input and the final step finished with FinishReasonStop.
	// Returns nil otherwise. Blocks until the stream is complete.
	Output func() any
}

// ResponseFormat configures how the model should format its response.
// Mirrors ai-sdk's LanguageModelV3CallOptions.responseFormat.
type ResponseFormat struct {
	// Type is "text" or "json".
	Type string `json:"type"`

	// Schema is the JSON schema for structured output (when Type is "json").
	Schema json.RawMessage `json:"schema,omitempty"`

	// Name is an optional name for the schema.
	Name string `json:"name,omitempty"`

	// Description is an optional description for the schema.
	Description string `json:"description,omitempty"`
}

// Output is the interface for output strategies (text, object, array, choice, json).
// Implementations live in the goai/output package.
type Output interface {
	// Name returns the name of the output mode.
	Name() string

	// ResponseFormat returns the response format configuration for the model.
	ResponseFormat() *ResponseFormat

	// ParseComplete parses the complete model output.
	ParseComplete(text string, ctx OutputParseContext) (any, error)

	// ParsePartial parses partial output during streaming.
	// Returns nil if the partial output cannot be parsed.
	ParsePartial(text string) any
}

// OutputParseContext provides context for parsing complete output.
type OutputParseContext struct {
	FinishReason FinishReason
	Usage        Usage
}

// ToolCall represents a tool call from the model.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// StepResultData is the minimal interface for step results needed by callbacks.
// The full StepResult type is defined in the main goai package.
type StepResultData interface {
	Text() string
	ToolCalls() []ToolCall
	ToolResults() []ToolResultEvent
	GetFinishReason() FinishReason
	GetUsage() Usage
}

// Input represents the input to StreamText.
type Input struct {
	// Model is the language model to use.
	Model Model

	// Messages is the conversation history.
	Messages []message.Message

	// Tools is the set of available tools (user-facing).
	// The core converts this to an ordered slice before passing to providers.
	Tools tool.Set

	// ActiveTools limits which tools the model can use (optional).
	ActiveTools []string

	// ToolChoice controls how the model selects tools.
	// Can be "auto", "none", "required", or a specific tool name.
	// Matches ai-sdk's toolChoice parameter.
	ToolChoice any

	// Temperature controls randomness (0-2).
	Temperature *float64

	// TopP controls nucleus sampling (0-1).
	TopP *float64

	// TopK controls top-k sampling.
	TopK *int

	// MaxOutputTokens limits the response length.
	MaxOutputTokens *int

	// StopSequences are strings that stop generation.
	StopSequences []string

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// MaxRetries is the number of retry attempts.
	MaxRetries int

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Output is the optional output strategy for parsing the model's response.
	// When set, ResponseFormat is sent to the model on every step, and the
	// final step's text is parsed via Output.ParseComplete (only when the
	// final step finishes with FinishReasonStop).
	Output Output

	// OnError is called when an error occurs.
	OnError func(error)

	// RepairToolCall attempts to fix malformed tool calls.
	RepairToolCall func(failed FailedToolCall) (*RepairedToolCall, error)

	// --- Multi-step tool loop options ---

	// MaxSteps is the maximum number of tool-calling steps (default 1).
	// When > 1, tools are executed automatically and the model is called again
	// until a stop condition is met or MaxSteps is reached.
	MaxSteps int

	// OnStepFinish is called after each step completes.
	// Receives the step result with all content, tool calls, and tool results.
	OnStepFinish func(step StepResultData)

	// OnFinish is called after all steps complete.
	// Receives the final step result plus all accumulated steps and total usage.
	OnFinish func(result OnFinishData)

	// Executor handles tool execution. If nil, tools are executed locally
	// using their Execute functions. Set this to use remote execution.
	Executor tool.Executor
}

// OnFinishData contains data passed to the OnFinish callback.
type OnFinishData struct {
	// Steps contains all step results from the generation.
	Steps []StepResultData
	// TotalUsage is the accumulated token usage across all steps.
	TotalUsage Usage
	// FinalStep is the last step result.
	FinalStep StepResultData
}

// CallOptions is the provider-facing input (like ai-sdk's LanguageModelV3CallOptions).
// This is what providers receive - it has Tools as an already-ordered slice.
type CallOptions struct {
	// Messages is the conversation history.
	Messages []message.Message `json:"messages,omitempty"`

	// Tools is the ordered list of tools (already prepared by the core).
	Tools []tool.Tool `json:"tools,omitempty"`

	// ToolChoice controls how the model selects tools.
	ToolChoice any `json:"toolChoice,omitempty"`

	// Temperature controls randomness (0-2).
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls nucleus sampling (0-1).
	TopP *float64 `json:"topP,omitempty"`

	// TopK controls top-k sampling.
	TopK *int `json:"topK,omitempty"`

	// PresencePenalty affects the likelihood of the model to repeat
	// information already in the prompt. Mirrors ai-sdk's CallOptions.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`

	// FrequencyPenalty affects the likelihood of the model to repeatedly
	// use the same words or phrases. Mirrors ai-sdk's CallOptions.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`

	// Seed requests deterministic sampling when supported by the provider.
	// Mirrors ai-sdk's CallOptions.
	Seed *int `json:"seed,omitempty"`

	// MaxOutputTokens limits the response length.
	MaxOutputTokens *int `json:"maxOutputTokens,omitempty"`

	// StopSequences are strings that stop generation.
	StopSequences []string `json:"stopSequences,omitempty"`

	// AbortSignal allows cancellation (not serializable).
	AbortSignal context.Context `json:"-"`

	// Headers are additional HTTP headers.
	Headers map[string]string `json:"headers,omitempty"`

	// ResponseFormat configures structured output (text or json with optional schema).
	ResponseFormat *ResponseFormat `json:"responseFormat,omitempty"`

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any `json:"providerOptions,omitempty"`
}

// Model represents a language model.
type Model interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// Stream sends a request and returns a stream of events.
	Stream(ctx context.Context, options *CallOptions) (<-chan Event, error)
}

// FailedToolCall contains information about a failed tool call.
type FailedToolCall struct {
	ToolCallID string
	ToolName   string
	Input      json.RawMessage
	Error      error
}

// RepairedToolCall contains the repaired tool call.
type RepairedToolCall struct {
	ToolName string
	Input    json.RawMessage
}
