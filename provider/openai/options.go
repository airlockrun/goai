package openai

// AllowedTools restricts which tools the model may call without removing any
// from the request, preserving prompt caching. Mode is "auto" (default) or
// "required". ai-sdk #15038.
type AllowedTools struct {
	ToolNames []string `json:"toolNames"`
	Mode      string   `json:"mode,omitempty"`
}

// ResponsesOptions contains provider-specific options for the OpenAI Responses API.
// These options match ai-sdk's OpenAIResponsesProviderOptions schema.
// See: ai-sdk/packages/openai/src/responses/openai-responses-options.ts
type ResponsesOptions struct {
	// Conversation is the ID of the OpenAI Conversation to continue.
	// You must create a conversation first via the OpenAI API.
	// Cannot be used in conjunction with PreviousResponseID.
	Conversation string `json:"conversation,omitempty"`

	// Include specifies extra fields to include in the response.
	// Example values: "reasoning.encrypted_content", "file_search_call.results", "message.output_text.logprobs"
	Include []string `json:"include,omitempty"`

	// Instructions for the model.
	// They can be used to change the system or developer message when continuing
	// a conversation using the PreviousResponseID option.
	Instructions string `json:"instructions,omitempty"`

	// Logprobs returns the log probabilities of the tokens.
	// Can be true (return logprobs) or a number 1-20 (return top N logprobs).
	// Including logprobs increases response size and can slow down response times.
	Logprobs any `json:"logprobs,omitempty"`

	// MaxToolCalls is the maximum number of total calls to built-in tools
	// that can be processed in a response. This applies across all built-in
	// tool calls, not per individual tool.
	MaxToolCalls *int `json:"maxToolCalls,omitempty"`

	// Metadata is additional metadata to store with the generation.
	Metadata any `json:"metadata,omitempty"`

	// ParallelToolCalls controls whether to use parallel tool calls.
	// Defaults to true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// AllowedTools restricts the callable tools to a subset while keeping the
	// full tools list intact, so prompt caching is preserved across requests
	// with different allowlists. When set, it overrides the request-level
	// ToolChoice and emits tool_choice: {type: "allowed_tools", mode, tools}.
	// ai-sdk #15038.
	AllowedTools *AllowedTools `json:"allowedTools,omitempty"`

	// PreviousResponseID is the ID of the previous response for conversation continuation.
	PreviousResponseID string `json:"previousResponseId,omitempty"`

	// PromptCacheKey sets a cache key to tie this prompt to cached prefixes
	// for better caching performance.
	PromptCacheKey string `json:"promptCacheKey,omitempty"`

	// PromptCacheRetention is the retention policy for the prompt cache.
	// Values: "in_memory" (default), "24h" (extended, only for 5.1 series models)
	PromptCacheRetention string `json:"promptCacheRetention,omitempty"`

	// ReasoningEffort controls reasoning effort for reasoning models.
	// Values: "none", "minimal", "low", "medium", "high", "xhigh"
	// Note: "none" is only for GPT-5.1 models, "xhigh" only for GPT-5.1-Codex-Max.
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// ReasoningSummary controls reasoning summary output from the model.
	// Values: "auto" (automatically receive richest level), "detailed" (comprehensive summaries)
	ReasoningSummary string `json:"reasoningSummary,omitempty"`

	// SafetyIdentifier is the identifier for safety monitoring and tracking.
	SafetyIdentifier string `json:"safetyIdentifier,omitempty"`

	// ServiceTier is the service tier for the request.
	// Values: "auto" (default), "flex" (50% cheaper, higher latency), "priority" (faster, Enterprise), "default"
	ServiceTier string `json:"serviceTier,omitempty"`

	// Store controls whether to store the generation. Defaults to true.
	Store *bool `json:"store,omitempty"`

	// PassThroughUnsupportedFiles forwards non-image inline file parts as
	// generic input files. Inline file inputs are otherwise restricted to
	// images and PDFs; enable this when the target model accepts additional
	// media types such as text/csv. ai-sdk #15297.
	PassThroughUnsupportedFiles bool `json:"passThroughUnsupportedFiles,omitempty"`

	// StrictJsonSchema controls whether to use strict JSON schema validation.
	// Defaults to true.
	StrictJsonSchema *bool `json:"strictJsonSchema,omitempty"`

	// TextVerbosity controls the verbosity of the model's responses.
	// Values: "low" (concise), "medium", "high" (verbose)
	TextVerbosity string `json:"textVerbosity,omitempty"`

	// Truncation controls output truncation.
	// Values: "auto" (default, truncates automatically), "disabled" (no truncation)
	Truncation string `json:"truncation,omitempty"`

	// User is a unique identifier representing your end-user for abuse monitoring.
	User string `json:"user,omitempty"`

	// SystemMessageMode overrides how system messages are handled.
	// Values: "system" (default for most models), "developer" (for reasoning models), "remove"
	SystemMessageMode string `json:"systemMessageMode,omitempty"`

	// ForceReasoning forces treating this model as a reasoning model.
	// Useful for "stealth" reasoning models via custom baseURL where the model ID
	// is not recognized by the SDK's allowlist.
	ForceReasoning bool `json:"forceReasoning,omitempty"`
}

// ChatOptions contains provider-specific options for the OpenAI Chat Completions API.
// These options match ai-sdk's OpenAIChatProviderOptions schema.
// See: ai-sdk/packages/openai/src/chat/openai-chat-options.ts
type ChatOptions struct {
	// Logprobs returns the log probabilities of the tokens.
	// Can be true (return logprobs) or a number 1-20 (return top N logprobs).
	Logprobs any `json:"logprobs,omitempty"`

	// LogitBias modifies the likelihood of specified tokens appearing in the completion.
	// Maps token IDs to bias values from -100 to 100.
	LogitBias map[string]int `json:"logitBias,omitempty"`

	// ParallelToolCalls controls whether to use parallel tool calls.
	// Defaults to true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// PresencePenalty penalizes new tokens based on whether they appear in the text so far.
	// Range: -2.0 to 2.0. Positive values increase likelihood to talk about new topics.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`

	// FrequencyPenalty penalizes new tokens based on their frequency in the text so far.
	// Range: -2.0 to 2.0. Positive values decrease likelihood to repeat the same line.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`

	// ReasoningEffort controls reasoning effort for reasoning models.
	// Values: "low", "medium", "high"
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// Store controls whether to store the generation.
	Store *bool `json:"store,omitempty"`

	// StrictJsonSchema controls whether to use strict JSON schema validation.
	StrictJsonSchema *bool `json:"strictJsonSchema,omitempty"`

	// StructuredOutputs controls whether to use structured outputs.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// User is a unique identifier representing your end-user for abuse monitoring.
	User string `json:"user,omitempty"`
}
