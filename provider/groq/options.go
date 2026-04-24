package groq

// ChatOptions contains provider-specific options for the Groq API.
// These options match ai-sdk's GroqProviderOptions schema.
// See: ai-sdk/packages/groq/src/groq-chat-options.ts
//
// Note: Groq uses an OpenAI-compatible API through the openaicompat package.
// These options define what Groq supports, but wiring them to the request
// requires updating the openaicompat implementation.
type ChatOptions struct {
	// ReasoningFormat controls how reasoning is returned.
	// Values: "parsed" (structured), "raw" (as-is), "hidden" (omitted)
	ReasoningFormat string `json:"reasoningFormat,omitempty"`

	// ReasoningEffort specifies the reasoning effort level for model inference.
	// Values: "none", "default", "low", "medium", "high"
	// See: https://console.groq.com/docs/reasoning#reasoning-effort
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// ParallelToolCalls controls whether to enable parallel function calling
	// during tool use. Default is true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// User is a unique identifier representing your end-user for abuse monitoring.
	User string `json:"user,omitempty"`

	// StructuredOutputs controls whether to use structured outputs. Default is true.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// StrictJsonSchema controls whether to use strict JSON schema validation.
	// When true, the model uses constrained decoding to guarantee schema compliance.
	// Only used when structured outputs are enabled and a schema is provided.
	// Default is true.
	StrictJsonSchema *bool `json:"strictJsonSchema,omitempty"`

	// ServiceTier is the service tier for the request.
	// Values: "on_demand" (default), "flex" (higher throughput), "auto"
	// (uses on_demand, falls back to flex), "performance" (ai-sdk #cb3ca8f).
	ServiceTier string `json:"serviceTier,omitempty"`
}
