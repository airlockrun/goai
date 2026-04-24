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
}

// ThinkingConfig configures DeepSeek's thinking behavior.
type ThinkingConfig struct {
	// Type can be "enabled" or "disabled". Default is "enabled".
	Type string `json:"type,omitempty"`
}
