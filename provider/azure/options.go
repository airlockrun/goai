package azure

// ChatOptions contains provider-specific options for the Azure OpenAI API.
// Azure OpenAI uses the same options as OpenAI.
// See: ai-sdk/packages/openai/src/responses/openai-responses-options.ts
//
// Note: Azure OpenAI is essentially OpenAI with a different endpoint.
// Use the same options as the OpenAI provider.
type ChatOptions struct {
	// ReasoningEffort controls reasoning effort for reasoning models.
	// Values: "none", "minimal", "low", "medium", "high", "xhigh"
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// ReasoningSummary controls reasoning summary output from the model.
	// Values: "auto", "detailed"
	ReasoningSummary string `json:"reasoningSummary,omitempty"`

	// Store controls whether to store the generation. Defaults to true.
	Store *bool `json:"store,omitempty"`

	// StrictJsonSchema controls whether to use strict JSON schema validation.
	StrictJsonSchema *bool `json:"strictJsonSchema,omitempty"`

	// User is a unique identifier representing your end-user for abuse monitoring.
	User string `json:"user,omitempty"`

	// ParallelToolCalls controls whether to use parallel tool calls.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`
}
