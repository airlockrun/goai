package mistral

// ChatOptions contains provider-specific options for the Mistral API.
// These options match ai-sdk's MistralLanguageModelOptions schema.
// See: ai-sdk/packages/mistral/src/mistral-chat-options.ts
//
// Note: Mistral uses an OpenAI-compatible API through the openaicompat package.
// These options define what Mistral supports, but wiring them to the request
// requires updating the openaicompat implementation.
type ChatOptions struct {
	// SafePrompt controls whether to inject a safety prompt before all conversations.
	// Default is false.
	SafePrompt *bool `json:"safePrompt,omitempty"`

	// DocumentImageLimit limits the number of images in document processing.
	DocumentImageLimit int `json:"documentImageLimit,omitempty"`

	// DocumentPageLimit limits the number of pages in document processing.
	DocumentPageLimit int `json:"documentPageLimit,omitempty"`

	// StructuredOutputs controls whether to use structured outputs. Default is true.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// StrictJsonSchema controls whether to use strict JSON schema validation.
	// Default is false.
	StrictJsonSchema *bool `json:"strictJsonSchema,omitempty"`

	// ParallelToolCalls controls whether to enable parallel function calling
	// during tool use. When false, the model uses at most one tool per response.
	// Default is true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// ReasoningEffort toggles reasoning on models that support adjustable
	// reasoning such as mistral-small-latest and mistral-small-2603.
	// Values: "high" (enable), "none" (disable). (ai-sdk #297e685)
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
}
