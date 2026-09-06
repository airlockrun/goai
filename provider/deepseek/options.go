package deepseek

// ChatOptions contains provider-specific options for the DeepSeek API.
// These options match ai-sdk's DeepSeekChatOptions schema.
// See: ai-sdk/packages/deepseek/src/chat/deepseek-chat-options.ts
//
// DeepSeek uses the OpenAI-compatible transport with model-specific normalization.
type ChatOptions struct {
	// Thinking configures the thinking/reasoning behavior.
	// Default is enabled.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`

	// ReasoningEffort controls the thinking strength for DeepSeek V4
	// reasoning models. Canonical values are "low", "high", and "max".
	// "medium" maps to "high" and "xhigh" maps to "max" with a warning.
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
}

// ThinkingConfig configures DeepSeek's thinking behavior.
type ThinkingConfig struct {
	// Type can be "enabled" or "disabled". "adaptive" maps to "enabled"
	// with a warning. Default is "enabled". See
	// https://api-docs.deepseek.com/guides/thinking_mode.
	Type string `json:"type,omitempty"`
}
