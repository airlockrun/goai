package cohere

// ChatOptions contains provider-specific options for the Cohere API.
// These options match ai-sdk's CohereChatModelOptions schema.
// See: ai-sdk/packages/cohere/src/cohere-chat-options.ts
type ChatOptions struct {
	// Thinking configures reasoning features.
	// See: https://docs.cohere.com/reference/chat#request.body.thinking
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// EmbeddingOptions contains provider-specific options for Cohere's
// Embed API (ai-sdk CohereEmbeddingOptions).
type EmbeddingOptions struct {
	// InputType overrides the default "search_document". Values:
	// "search_document", "search_query", "classification", "clustering".
	InputType string `json:"inputType,omitempty"`

	// OutputDimension selects the embedding dimensionality. Only the
	// embed-v4 family supports this; Cohere accepts 256, 512, 1024,
	// 1536 (ai-sdk #0df64d6).
	OutputDimension int `json:"outputDimension,omitempty"`
}

// ThinkingConfig configures Cohere's reasoning features.
type ThinkingConfig struct {
	// Type can be "enabled" or "disabled". Default is "enabled".
	Type string `json:"type,omitempty"`

	// TokenBudget is the maximum number of tokens the model can use for thinking.
	// Must be a positive integer. The model will stop thinking if it reaches
	// the budget and proceed with the response.
	TokenBudget int `json:"tokenBudget,omitempty"`
}
