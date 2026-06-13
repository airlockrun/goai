package deepseek

// ChatOptions contains provider-specific options for the DeepSeek API.
// These options match ai-sdk's DeepSeekChatOptions schema.
// See: ai-sdk/packages/deepseek/src/chat/deepseek-chat-options.ts
//
// Note: DeepSeek uses an OpenAI-compatible API through the openaicompat package.
// These options define what DeepSeek supports, but wiring them to the request
// requires updating the openaicompat implementation.
type ChatOptions struct {
	// Thinking configures the thinking/reasoning behavior.
	// Default is enabled.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`

	// ReasoningEffort controls the thinking strength for DeepSeek V4
	// reasoning models. DeepSeek's API accepts "low", "medium", "high",
	// "xhigh", and "max"; it maps "low"/"medium" to "high" and "xhigh" to
	// "max" server-side for compatibility with other providers.
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
}

// ThinkingConfig configures DeepSeek's thinking behavior.
type ThinkingConfig struct {
	// Type can be "adaptive", "enabled", or "disabled". "adaptive" lets the
	// model decide when to think. Default is "enabled". See
	// https://api-docs.deepseek.com/guides/thinking_mode.
	Type string `json:"type,omitempty"`
}
