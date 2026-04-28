package anthropic

// MessagesOptions contains provider-specific options for the Anthropic Messages API.
// These options match ai-sdk's AnthropicProviderOptions schema.
// See: ai-sdk/packages/anthropic/src/anthropic-messages-options.ts
type MessagesOptions struct {
	// SendReasoning controls whether to send reasoning to the model.
	// This allows you to deactivate reasoning inputs for models that do not support them.
	SendReasoning *bool `json:"sendReasoning,omitempty"`

	// StructuredOutputMode determines how structured outputs are generated.
	// Values: "outputFormat" (use output_format parameter), "jsonTool" (use special json tool), "auto" (default)
	StructuredOutputMode string `json:"structuredOutputMode,omitempty"`

	// Thinking configures Claude's extended thinking.
	// When enabled, responses include thinking content blocks showing Claude's thinking process.
	// Requires minimum 1,024 tokens budget and counts towards max_tokens limit.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`

	// DisableParallelToolUse disables parallel function calling during tool use.
	// When true, Claude will use at most one tool per response. Default is false.
	DisableParallelToolUse *bool `json:"disableParallelToolUse,omitempty"`

	// MCPServers to be utilized in this request.
	MCPServers []MCPServer `json:"mcpServers,omitempty"`

	// Container configures Agent Skills (document processing, data analysis).
	// Skills enable Claude to perform specialized tasks like PPTX, DOCX, PDF, XLSX processing.
	// Requires code execution tool to be enabled.
	Container *ContainerConfig `json:"container,omitempty"`

	// ToolStreaming controls whether to enable tool streaming.
	// When false, the model returns all tool calls at once after a delay. Default is true.
	ToolStreaming *bool `json:"toolStreaming,omitempty"`

	// Effort advises the model how much effort to spend on the task. Lands
	// at output_config.effort on the wire (ai-sdk parity). Suppressed when
	// Thinking.Type is explicitly "disabled". Unset by default.
	// Values: "low", "medium", "high", "xhigh", "max".
	Effort string `json:"effort,omitempty"`

	// ContextManagement configures context management edits.
	ContextManagement *ContextManagement `json:"contextManagement,omitempty"`

	// CacheControl configures prompt caching at the request level. Mirrors
	// ai-sdk PR #17978c6 which lets callers enable auto-caching by setting
	// providerOptions.anthropic.cacheControl directly.
	CacheControl *CacheControlConfig `json:"cacheControl,omitempty"`

	// Metadata carries request-level metadata, notably `metadata.user_id`
	// (ai-sdk #05b8ca2).
	Metadata *MessagesMetadata `json:"metadata,omitempty"`

	// TaskBudget advises the model of the total token budget available for
	// the current agentic task. Advisory only — does not enforce a hard
	// limit.
	TaskBudget *TaskBudget `json:"taskBudget,omitempty"`

	// Speed selects the inference speed profile. Only "fast" and "standard"
	// are supported; fast mode is currently restricted to claude-opus-4-6
	// (ai-sdk #0a0d29c).
	Speed string `json:"speed,omitempty"`

	// InferenceGeo controls where model inference runs. "us" or "global"
	// (ai-sdk #61f1a61).
	InferenceGeo string `json:"inferenceGeo,omitempty"`

	// AnthropicBeta is a list of beta feature flags to enable via the
	// `anthropic-beta` HTTP header (ai-sdk #e49c34d). Use only when you
	// need a beta that goai doesn't set automatically.
	AnthropicBeta []string `json:"anthropicBeta,omitempty"`
}

// CacheControlConfig mirrors ai-sdk's cacheControl option at the request
// level. TTL is "5m" (default) or "1h".
type CacheControlConfig struct {
	Type string `json:"type"` // "ephemeral"
	TTL  string `json:"ttl,omitempty"`
}

// MessagesMetadata carries request-level metadata. `UserID` must not be
// PII per Anthropic's policy — use a UUID or opaque hash.
type MessagesMetadata struct {
	UserID string `json:"userId,omitempty"`
}

// TaskBudget advises the model of the total token budget available for
// the current agentic turn (ai-sdk packages/anthropic/.../taskBudget).
type TaskBudget struct {
	Type      string `json:"type"` // "tokens"
	Total     int    `json:"total"`
	Remaining int    `json:"remaining,omitempty"`
}

// ThinkingConfig configures Claude's extended thinking capability.
type ThinkingConfig struct {
	// Type is "adaptive" (Sonnet 4.6, Opus 4.6, and newer), "enabled"
	// (Sonnet 4.5 and earlier, Opus 4.5 and earlier), or "disabled".
	Type string `json:"type"`

	// BudgetTokens is the token budget for thinking. Minimum 1,024 tokens.
	// Used when Type is "enabled".
	BudgetTokens int `json:"budgetTokens,omitempty"`

	// Display controls whether thinking content is included in the response.
	// Only applies when Type is "adaptive". Values: "omitted" (default for
	// Opus 4.7+) or "summarized" (required to see reasoning output).
	Display string `json:"display,omitempty"`
}

// MCPServer represents an MCP server configuration.
type MCPServer struct {
	// Type must be "url".
	Type string `json:"type"`

	// Name of the MCP server.
	Name string `json:"name"`

	// URL of the MCP server.
	URL string `json:"url"`

	// AuthorizationToken for the MCP server.
	AuthorizationToken string `json:"authorizationToken,omitempty"`

	// ToolConfiguration for the MCP server.
	ToolConfiguration *ToolConfiguration `json:"toolConfiguration,omitempty"`
}

// ToolConfiguration for MCP server.
type ToolConfiguration struct {
	// Enabled controls if tools are enabled.
	Enabled *bool `json:"enabled,omitempty"`

	// AllowedTools is a list of allowed tool names.
	AllowedTools []string `json:"allowedTools,omitempty"`
}

// ContainerConfig for Agent Skills.
type ContainerConfig struct {
	// ID of the container.
	ID string `json:"id,omitempty"`

	// Skills to enable.
	Skills []Skill `json:"skills,omitempty"`
}

// Skill represents an Agent Skill.
type Skill struct {
	// Type is either "anthropic" or "custom".
	Type string `json:"type"`

	// SkillID is the skill identifier.
	SkillID string `json:"skillId"`

	// Version of the skill.
	Version string `json:"version,omitempty"`
}

// ContextManagement configures context management edits.
type ContextManagement struct {
	// Edits is a list of context management edits.
	Edits []ContextEdit `json:"edits"`
}

// ContextEdit represents a context management edit.
// This is a simplified representation - the full ai-sdk schema has discriminated unions.
type ContextEdit struct {
	// Type is "clear_tool_uses_20250919", "clear_thinking_20251015", or
	// "compact_20260112" (ai-sdk #c60b393).
	Type string `json:"type"`

	// Trigger for when to apply the edit (for clear_tool_uses_20250919 or
	// compact_20260112).
	Trigger *ContextTrigger `json:"trigger,omitempty"`

	// Keep configuration.
	Keep any `json:"keep,omitempty"`

	// ClearAtLeast configuration (for clear_tool_uses_20250919).
	ClearAtLeast *ClearAtLeast `json:"clearAtLeast,omitempty"`

	// ClearToolInputs controls whether to clear tool inputs (for clear_tool_uses_20250919).
	ClearToolInputs *bool `json:"clearToolInputs,omitempty"`

	// ExcludeTools is a list of tools to exclude from clearing (for clear_tool_uses_20250919).
	ExcludeTools []string `json:"excludeTools,omitempty"`

	// PauseAfterCompaction (compact_20260112 only): pause the agentic turn
	// after compaction completes so the caller can inspect state.
	PauseAfterCompaction *bool `json:"pauseAfterCompaction,omitempty"`

	// Instructions (compact_20260112 only): caller-supplied instructions
	// describing what to preserve during compaction.
	Instructions string `json:"instructions,omitempty"`
}

// ContextKeep declares what to preserve for a clear edit. Two shapes are
// supported to mirror ai-sdk's discriminated union:
//   - All=true serializes the literal string "all" (clear_thinking only).
//   - Otherwise serializes {type, value} — e.g. {type:"tool_uses", value:5}
//     or {type:"thinking_turns", value:2}.
type ContextKeep struct {
	All   bool
	Type  string
	Value int
}

// ContextTrigger for context management.
type ContextTrigger struct {
	// Type is either "input_tokens" or "tool_uses".
	Type string `json:"type"`

	// Value is the threshold value.
	Value int `json:"value"`
}

// ClearAtLeast configuration.
type ClearAtLeast struct {
	// Type must be "input_tokens".
	Type string `json:"type"`

	// Value is the minimum tokens to clear.
	Value int `json:"value"`
}

// FilePartOptions contains provider-specific options for individual file parts.
// These options match ai-sdk's AnthropicFilePartProviderOptions schema.
type FilePartOptions struct {
	// Citations configuration for this document.
	Citations *CitationsConfig `json:"citations,omitempty"`

	// Title is a custom title for the document. If not provided, the filename is used.
	Title string `json:"title,omitempty"`

	// Context about the document that will be passed to the model
	// but not used towards cited content.
	Context string `json:"context,omitempty"`
}

// CitationsConfig for document citations.
type CitationsConfig struct {
	// Enabled controls whether citations are enabled for this document.
	Enabled bool `json:"enabled"`
}

// CacheControl returns ProviderOptions that enable Anthropic prompt caching.
// Optional ttl: "5m" (default if omitted) or "1h".
//
// Usage:
//
//	message.Message{
//	    Role:            message.RoleSystem,
//	    Content:         message.Content{Text: "..."},
//	    ProviderOptions: anthropic.CacheControl(),
//	}
//
//	message.TextPart{
//	    Text:            "cached context",
//	    ProviderOptions: anthropic.CacheControl("1h"),
//	}
func CacheControl(ttl ...string) map[string]any {
	cc := map[string]any{"type": "ephemeral"}
	if len(ttl) > 0 && ttl[0] != "" {
		cc["ttl"] = ttl[0]
	}
	return map[string]any{
		"anthropic": map[string]any{
			"cacheControl": cc,
		},
	}
}
