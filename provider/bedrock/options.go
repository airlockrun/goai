package bedrock

// ChatOptions contains provider-specific options for the Amazon Bedrock API.
// These options match ai-sdk's BedrockProviderOptions schema.
// See: ai-sdk/packages/amazon-bedrock/src/bedrock-chat-options.ts
type ChatOptions struct {
	// AdditionalModelRequestFields contains additional inference parameters
	// that the model supports, beyond the base set in inferenceConfig.
	AdditionalModelRequestFields map[string]any `json:"additionalModelRequestFields,omitempty"`

	// ReasoningConfig configures reasoning/thinking behavior.
	ReasoningConfig *ReasoningConfig `json:"reasoningConfig,omitempty"`

	// AnthropicBeta lists Anthropic beta features to enable.
	AnthropicBeta []string `json:"anthropicBeta,omitempty"`

	// ServiceTier selects Bedrock's service tier (ai-sdk #df099b9).
	// Values are model-specific, e.g. "standard" or "priority".
	ServiceTier string `json:"serviceTier,omitempty"`

	// CacheControl enables prompt caching with an optional TTL
	// (ai-sdk #b128d9b). TTL is "5m" (default) or "1h".
	CacheControl *CacheControlConfig `json:"cacheControl,omitempty"`
}

// CacheControlConfig mirrors Anthropic's cache_control object at the
// request root.
type CacheControlConfig struct {
	Type string `json:"type"` // "ephemeral"
	TTL  string `json:"ttl,omitempty"`
}

// ReasoningConfig configures Bedrock reasoning behavior.
type ReasoningConfig struct {
	// Type can be "adaptive" (Sonnet 4.6+, Opus 4.6+), "enabled", or
	// "disabled". "adaptive" mirrors the Anthropic adaptive thinking
	// surface (ai-sdk #632ab10).
	Type string `json:"type,omitempty"`

	// BudgetTokens is the token budget for reasoning (when Type is
	// "enabled").
	BudgetTokens int `json:"budgetTokens,omitempty"`

	// MaxReasoningEffort is the maximum reasoning effort level.
	// Values: "low", "medium", "high", "xhigh", "max" (ai-sdk #632ab10
	// expanded the range).
	MaxReasoningEffort string `json:"maxReasoningEffort,omitempty"`
}

// FilePartOptions contains provider-specific options for individual file parts.
// These options match ai-sdk's BedrockFilePartProviderOptions schema.
type FilePartOptions struct {
	// Citations configuration for this document.
	Citations *CitationsConfig `json:"citations,omitempty"`
}

// CitationsConfig for document citations.
type CitationsConfig struct {
	// Enabled controls whether citations are enabled for this document.
	Enabled bool `json:"enabled"`
}
