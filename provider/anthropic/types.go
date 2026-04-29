package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Request types

type anthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        any                `json:"system,omitempty"` // string or []systemBlock with cache_control
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	// Tools is a mixed slice: function tools (anthropicTool) and
	// hosted/provider-defined tools (one of the typed structs in
	// hosted_tools.go) share this slice.
	Tools []any `json:"tools,omitempty"`

	// Thinking configures extended thinking (reasoning) for supported models.
	Thinking *anthropicThinking `json:"thinking,omitempty"`

	// ToolChoice controls tool selection. Values: "auto", "any", "none", or {"type": "tool", "name": "tool_name"}
	ToolChoice any `json:"tool_choice,omitempty"`

	// Metadata for the request
	Metadata *anthropicMetadata `json:"metadata,omitempty"`

	// MCPServers for MCP tool execution
	MCPServers []anthropicMCPServer `json:"mcp_servers,omitempty"`

	// Container for Agent Skills
	Container *anthropicContainer `json:"container,omitempty"`

	// ContextManagement declares server-side context-window compaction
	// policies (clear_tool_uses_20250919, clear_thinking_20251015,
	// compact_20260112).
	ContextManagement *anthropicContextManagement `json:"context_management,omitempty"`

	// CacheControl enables prompt caching at the request root
	// (ai-sdk #17978c6).
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`

	// Speed selects the inference speed profile ("fast" / "standard").
	Speed string `json:"speed,omitempty"`

	// InferenceGeo pins inference to a geography ("us" / "global").
	InferenceGeo string `json:"inference_geo,omitempty"`

	// TaskBudget is an advisory budget for the agentic turn.
	TaskBudget *anthropicTaskBudget `json:"task_budget,omitempty"`

	// OutputConfig is the structured-output payload when
	// structuredOutputMode == "outputFormat" (ai-sdk #d98d9ba migrated
	// this from `output_format` → `output_config.format`).
	OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
}

// anthropicTaskBudget is the wire shape for task_budget.
type anthropicTaskBudget struct {
	Type      string `json:"type"` // "tokens"
	Total     int    `json:"total"`
	Remaining int    `json:"remaining,omitempty"`
}

// anthropicOutputConfig carries output-related configuration. Currently used
// for structured-output (format) and the effort hint. ai-sdk colocates these
// (and task_budget) under output_config.
type anthropicOutputConfig struct {
	Format *anthropicOutputFormat `json:"format,omitempty"`

	// Effort advises the model how much effort to put into the task.
	// Values: "low" / "medium" / "high" / "xhigh" / "max". Mirrors
	// ai-sdk's anthropicOptions.effort which lands at output_config.effort
	// (anthropic-language-model.ts:454-456).
	Effort string `json:"effort,omitempty"`
}

type anthropicOutputFormat struct {
	Type   string          `json:"type"` // "json_schema"
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// anthropicContextManagement is the request-side payload. Mirrors ai-sdk's
// anthropic-messages-options.ts contextManagement schema.
type anthropicContextManagement struct {
	Edits []anthropicContextEdit `json:"edits"`
}

// anthropicContextEdit is one entry in context_management.edits. Only the
// fields matching the discriminant Type are populated.
type anthropicContextEdit struct {
	Type                 string                        `json:"type"` // "clear_tool_uses_20250919" | "clear_thinking_20251015" | "compact_20260112"
	Trigger              *anthropicContextTrigger      `json:"trigger,omitempty"`
	Keep                 *anthropicContextKeep         `json:"keep,omitempty"`
	ClearAtLeast         *anthropicContextClearAtLeast `json:"clear_at_least,omitempty"`
	ClearToolInputs      *bool                         `json:"clear_tool_inputs,omitempty"`
	ExcludeTools         []string                      `json:"exclude_tools,omitempty"`
	PauseAfterCompaction *bool                         `json:"pause_after_compaction,omitempty"`
	Instructions         string                        `json:"instructions,omitempty"`
}

// anthropicContextTrigger declares when a clear edit fires.
type anthropicContextTrigger struct {
	Type  string `json:"type"` // "input_tokens" | "tool_uses"
	Value int    `json:"value"`
}

// anthropicContextKeep declares what to retain. For clear_tool_uses the type
// is "tool_uses"; for clear_thinking the type is "thinking_turns" or the
// literal string "all" (serialized as a raw string below).
type anthropicContextKeep struct {
	// All=true serializes the literal string "all" (clear_thinking only).
	All bool `json:"-"`
	// Otherwise serializes an object {type, value}.
	Type  string `json:"type,omitempty"`
	Value int    `json:"value,omitempty"`
}

// MarshalJSON lets anthropicContextKeep emit either the string "all" or an
// object {type, value}, matching ai-sdk's union type.
func (k anthropicContextKeep) MarshalJSON() ([]byte, error) {
	if k.All {
		return json.Marshal("all")
	}
	return json.Marshal(struct {
		Type  string `json:"type"`
		Value int    `json:"value"`
	}{k.Type, k.Value})
}

// anthropicContextClearAtLeast sets the minimum amount to clear.
type anthropicContextClearAtLeast struct {
	Type  string `json:"type"` // "input_tokens"
	Value int    `json:"value"`
}

// anthropicContextAppliedEdits is the response-side payload, delivered in
// the message-stop / final message envelope under context_management.
type anthropicContextAppliedEdits struct {
	AppliedEdits []anthropicContextAppliedEdit `json:"applied_edits"`
}

type anthropicContextAppliedEdit struct {
	Type                 string `json:"type"`
	ClearedToolUses      int    `json:"cleared_tool_uses,omitempty"`
	ClearedThinkingTurns int    `json:"cleared_thinking_turns,omitempty"`
	ClearedInputTokens   int    `json:"cleared_input_tokens,omitempty"`
}

// anthropicMetadata contains request metadata
type anthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

// anthropicMCPServer represents an MCP server configuration
type anthropicMCPServer struct {
	Type               string                      `json:"type"` // "url"
	Name               string                      `json:"name"`
	URL                string                      `json:"url"`
	AuthorizationToken string                      `json:"authorization_token,omitempty"`
	ToolConfiguration  *anthropicToolConfiguration `json:"tool_configuration,omitempty"`
}

// anthropicToolConfiguration for MCP server
type anthropicToolConfiguration struct {
	Enabled      *bool    `json:"enabled,omitempty"`
	AllowedTools []string `json:"allowed_tools,omitempty"`
}

// anthropicContainer for Agent Skills
type anthropicContainer struct {
	ID     string           `json:"id,omitempty"`
	Skills []anthropicSkill `json:"skills,omitempty"`
}

// anthropicSkill represents an Agent Skill
type anthropicSkill struct {
	Type    string `json:"type"` // "anthropic" or "custom"
	SkillID string `json:"skill_id"`
	Version string `json:"version,omitempty"`
}

// systemBlock represents a system message block with cache control
type systemBlock struct {
	Type         string                 `json:"type"` // "text"
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl for prompt caching
type anthropicCacheControl struct {
	Type string `json:"type"`          // "ephemeral"
	TTL  string `json:"ttl,omitempty"` // "5m", "1h"
}

// anthropicThinking configures Claude's extended thinking.
type anthropicThinking struct {
	Type         string `json:"type"`                    // "adaptive" | "enabled" | "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"` // Minimum 1024 (when Type == "enabled")
	Display      string `json:"display,omitempty"`       // "omitted" | "summarized" (when Type == "adaptive")
}

type anthropicMessage struct {
	Role    string                  `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`

	// For text blocks
	Text string `json:"text,omitempty"`

	// For image and document blocks
	Source *anthropicSource `json:"source,omitempty"`

	// For document blocks
	Title     string              `json:"title,omitempty"`
	Context   string              `json:"context,omitempty"`
	Citations *anthropicCitations `json:"citations,omitempty"`

	// For tool_use blocks
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// For tool_result blocks
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`

	// Prompt caching
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCitations struct {
	Enabled bool `json:"enabled"`
}

type anthropicSource struct {
	Type      string `json:"type"`                 // "base64" | "text" | "url"
	MediaType string `json:"media_type,omitempty"` // required for base64/text, absent for url
	Data      string `json:"data,omitempty"`       // required for base64/text
	URL       string `json:"url,omitempty"`        // required for url sources
}

type anthropicTool struct {
	Name          string                  `json:"name"`
	Description   string                  `json:"description,omitempty"`
	InputSchema   json.RawMessage         `json:"input_schema"`
	InputExamples []anthropicInputExample `json:"input_examples,omitempty"`
	CacheControl  *anthropicCacheControl  `json:"cache_control,omitempty"`
}

type anthropicInputExample struct {
	Input json.RawMessage `json:"input"`
}

// Streaming response types

type anthropicStreamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index,omitempty"`

	// For message_start
	Message *anthropicMessageStart `json:"message,omitempty"`

	// For content_block_start
	ContentBlock *anthropicContentBlockStart `json:"content_block,omitempty"`

	// For content_block_delta and message_delta
	Delta *anthropicDelta `json:"delta,omitempty"`

	// Usage (message_delta) — typed for the fields we care about.
	Usage *anthropicUsageDelta `json:"usage,omitempty"`

	// UsageRaw holds the same payload as Usage but as raw JSON so we
	// can forward unknown/provider-specific usage fields to consumers
	// via providerMetadata.anthropic.usage (ai-sdk #8c2b1e1).
	UsageRaw json.RawMessage `json:"-"`
}

// UnmarshalJSON captures the raw `usage` object alongside the typed
// parse so downstream surfaces can forward it verbatim.
func (e *anthropicStreamEvent) UnmarshalJSON(data []byte) error {
	type alias anthropicStreamEvent
	aux := struct {
		*alias
	}{alias: (*alias)(e)}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	// Also capture the raw `usage` payload — the typed Usage above only
	// records known fields; UsageRaw preserves unknown ones for
	// providerMetadata forwarding.
	var envelope struct {
		Usage json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(data, &envelope); err == nil {
		e.UsageRaw = envelope.Usage
	}
	return nil
}

type anthropicMessageStart struct {
	ID    string              `json:"id"`
	Type  string              `json:"type"`
	Role  string              `json:"role"`
	Model string              `json:"model"`
	Usage *anthropicUsageInfo `json:"usage,omitempty"`
}

// anthropicUsageInfo carries the full raw usage object plus the known
// token fields. ai-sdk's anthropicMessagesUsageSchema expanded to
// include cache-token and modality-breakdown fields; rather than type
// each one separately we keep the typed fields plus the raw map so
// downstream consumers see exactly what Anthropic returned
// (ai-sdk PRs #b9d105f, #2445da4, #8c2b1e1).
type anthropicUsageInfo struct {
	InputTokens              int                 `json:"input_tokens"`
	OutputTokens             int                 `json:"output_tokens"`
	CacheCreationInputTokens int                 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                 `json:"cache_read_input_tokens,omitempty"`
	OutputTokensByType       *outputTokensByType `json:"output_tokens_by_type,omitempty"`
}

// outputTokensByType breaks OutputTokens down by text vs reasoning so
// callers can surface `outputTokens.text` (ai-sdk #2445da4).
type outputTokensByType struct {
	Text      int `json:"text,omitempty"`
	Reasoning int `json:"reasoning,omitempty"`
}

type anthropicContentBlockStart struct {
	Type string `json:"type"`

	// For text blocks
	Text string `json:"text,omitempty"`

	// For tool_use blocks
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

type anthropicDelta struct {
	Type string `json:"type"`

	// For text_delta
	Text string `json:"text,omitempty"`

	// For input_json_delta
	PartialJSON string `json:"partial_json,omitempty"`

	// For compaction_delta: server-side conversation summary that
	// ships after a compact_20260112 policy fires. ai-sdk #b094c07:
	// the server may send content:null on the first delta, so this
	// field is a *string — nil means "no content in this frame".
	Content *string `json:"content,omitempty"`

	// For message_delta
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`

	// For message_delta: server-side context_management results.
	ContextManagement *anthropicContextAppliedEdits `json:"context_management,omitempty"`
}

type anthropicUsageDelta struct {
	InputTokens              int                 `json:"input_tokens,omitempty"`
	OutputTokens             int                 `json:"output_tokens"`
	CacheCreationInputTokens int                 `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                 `json:"cache_read_input_tokens,omitempty"`
	OutputTokensByType       *outputTokensByType `json:"output_tokens_by_type,omitempty"`
}

// Conversion functions

func getTextFromContent(content message.Content) string {
	if content.Text != "" {
		return content.Text
	}
	for _, part := range content.Parts {
		if tp, ok := part.(message.TextPart); ok {
			return tp.Text
		}
	}
	return ""
}

// toolMessageToBlocks converts one goai tool-role message into the slice of
// tool_result content blocks it contributes to the surrounding user block.
// The user-block walker in BuildRequestBody calls this and appends the result
// alongside any sibling user-role parts so all tool_results for a turn land
// in a single anthropic user message — Anthropic's API requires this pairing.
func toolMessageToBlocks(msg message.Message) []anthropicContentBlock {
	// Collect non-ToolResult parts (attachments) from the message; goai
	// allows text/image/file parts to ride alongside ToolResults in a tool
	// message, and they get embedded inside each tool_result's content
	// array (legacy goai behavior preserved here).
	var attachments []message.Part
	for _, part := range msg.Content.Parts {
		switch part.(type) {
		case message.TextPart, message.ImagePart, message.FilePart:
			attachments = append(attachments, part)
		}
	}

	// Collect tool result parts to know which is last for cache-control.
	var toolResults []message.ToolResultPart
	for _, part := range msg.Content.Parts {
		if tr, ok := part.(message.ToolResultPart); ok {
			toolResults = append(toolResults, tr)
		}
	}

	out := make([]anthropicContentBlock, 0, len(toolResults))
	for i, tr := range toolResults {
		block := anthropicContentBlock{
			Type:      "tool_result",
			ToolUseID: tr.ToolCallID,
			IsError:   tr.IsError,
		}

		if len(attachments) > 0 {
			var contentBlocks []anthropicContentBlock
			resultStr := toolResultToString(tr.Result)
			if resultStr != "" {
				contentBlocks = append(contentBlocks, anthropicContentBlock{
					Type: "text",
					Text: resultStr,
				})
			}
			contentBlocks = append(contentBlocks, convertToAnthropicContent(message.Content{Parts: attachments}, nil)...)
			block.Content = contentBlocks
		} else {
			block.Content = toolResultToString(tr.Result)
		}

		// Part-level cache control, message-level fallback on last tool result.
		cc := getCacheControl(tr.ProviderOptions)
		if cc == nil && i == len(toolResults)-1 {
			cc = getCacheControl(msg.ProviderOptions)
		}
		block.CacheControl = cc

		out = append(out, block)
	}
	return out
}

// convertToolMessages wraps toolMessageToBlocks back into a one-element slice
// of anthropicMessage. Kept for the existing TestConvertToolMessages tests
// that exercise the block-construction logic directly. The live request
// builder no longer calls this — it calls toolMessageToBlocks and merges the
// blocks into the surrounding user block.
func convertToolMessages(msg message.Message) []anthropicMessage {
	blocks := toolMessageToBlocks(msg)
	if len(blocks) == 0 {
		return nil
	}
	return []anthropicMessage{{Role: "user", Content: blocks}}
}

func toolResultToString(result any) string {
	switch v := result.(type) {
	case string:
		return v
	default:
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
		return ""
	}
}

func convertToAnthropicContent(content message.Content, msgOpts map[string]any) []anthropicContentBlock {
	// If simple text, return single text block
	if content.Text != "" && len(content.Parts) == 0 {
		block := anthropicContentBlock{Type: "text", Text: content.Text}
		// Simple text is both first and last — apply message-level cache control.
		block.CacheControl = getCacheControl(msgOpts)
		return []anthropicContentBlock{block}
	}

	result := make([]anthropicContentBlock, 0, len(content.Parts))
	for i, part := range content.Parts {
		isLastPart := i == len(content.Parts)-1
		var partOpts map[string]any
		var block anthropicContentBlock

		switch p := part.(type) {
		case message.TextPart:
			partOpts = p.ProviderOptions
			block = anthropicContentBlock{
				Type: "text",
				Text: p.Text,
			}
		case message.ImagePart:
			partOpts = p.ProviderOptions
			src := &anthropicSource{Type: "base64", MediaType: p.MimeType, Data: p.Image}
			if strings.HasPrefix(p.Image, "http://") || strings.HasPrefix(p.Image, "https://") {
				src = &anthropicSource{Type: "url", URL: p.Image}
			}
			block = anthropicContentBlock{Type: "image", Source: src}
		case message.FilePart:
			partOpts = p.ProviderOptions
			if strings.HasPrefix(p.MimeType, "image/") {
				src := &anthropicSource{Type: "base64", MediaType: p.MimeType, Data: p.Data}
				if p.URL != "" {
					src = &anthropicSource{Type: "url", URL: p.URL}
				}
				block = anthropicContentBlock{Type: "image", Source: src}
			} else if p.MimeType == "application/pdf" {
				src := &anthropicSource{Type: "base64", MediaType: "application/pdf", Data: p.Data}
				if p.URL != "" {
					src = &anthropicSource{Type: "url", URL: p.URL}
				}
				block = anthropicContentBlock{
					Type:   "document",
					Source: src,
					Title:  p.Filename,
				}
			} else if p.MimeType == "text/plain" {
				src := &anthropicSource{Type: "text", MediaType: "text/plain", Data: p.Data}
				if p.URL != "" {
					src = &anthropicSource{Type: "url", URL: p.URL}
				}
				block = anthropicContentBlock{
					Type:   "document",
					Source: src,
					Title:  p.Filename,
				}
			} else {
				continue
			}
		default:
			continue
		}

		// Part-level cache control, with message-level fallback on last part.
		cc := getCacheControl(partOpts)
		if cc == nil && isLastPart {
			cc = getCacheControl(msgOpts)
		}
		block.CacheControl = cc

		result = append(result, block)
	}
	return result
}

func convertAssistantContent(content message.Content, msgOpts map[string]any) []anthropicContentBlock {
	result := make([]anthropicContentBlock, 0)

	// Collect all blocks first to determine which is last.
	type blockInfo struct {
		block    anthropicContentBlock
		partOpts map[string]any
	}
	var blocks []blockInfo

	// Add text if present
	text := getTextFromContent(content)
	if text != "" {
		blocks = append(blocks, blockInfo{
			block: anthropicContentBlock{Type: "text", Text: text},
		})
	}

	// Add tool calls
	for _, part := range content.Parts {
		if tc, ok := part.(message.ToolCallPart); ok {
			blocks = append(blocks, blockInfo{
				block: anthropicContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Name,
					Input: tc.Input,
				},
				partOpts: tc.ProviderOptions,
			})
		}
	}

	for i, bi := range blocks {
		isLastPart := i == len(blocks)-1
		cc := getCacheControl(bi.partOpts)
		if cc == nil && isLastPart {
			cc = getCacheControl(msgOpts)
		}
		bi.block.CacheControl = cc
		result = append(result, bi.block)
	}

	return result
}

func convertToAnthropicTools(tools []tool.Tool) ([]any, []string) {
	result, betas, _ := convertToAnthropicToolsWithWarnings(tools)
	return result, betas
}

// convertToAnthropicToolsWithWarnings is the warnings-aware variant. Caller
// receives an Unsupported warning for each unknown provider-defined tool so
// the drop is visible in the stream surface.
func convertToAnthropicToolsWithWarnings(tools []tool.Tool) ([]any, []string, []stream.Warning) {
	result := make([]any, 0, len(tools))
	var betas []string
	var warnings []stream.Warning

	for _, t := range tools {
		if t.Type == "provider" {
			hosted, betaHeader, ok := ConvertProviderTool(t)
			if !ok {
				warnings = append(warnings, stream.UnsupportedWarning(
					"tool",
					fmt.Sprintf("provider-defined tool %q is not supported by Anthropic", t.ProviderID),
				))
				continue
			}
			result = append(result, hosted)
			if betaHeader != "" {
				betas = append(betas, betaHeader)
			}
			continue
		}

		var examples []anthropicInputExample
		if len(t.InputExamples) > 0 {
			examples = make([]anthropicInputExample, len(t.InputExamples))
			for i, ex := range t.InputExamples {
				examples[i] = anthropicInputExample{Input: ex.Input}
			}
		}
		result = append(result, anthropicTool{
			Name:          t.Name,
			Description:   t.Description,
			InputSchema:   t.InputSchema,
			InputExamples: examples,
			CacheControl:  getCacheControl(t.ProviderOptions),
		})
	}

	return result, betas, warnings
}

// getCacheControl extracts Anthropic cache control from ProviderOptions.
// Accepts both "cacheControl" and "cache_control" keys for ai-sdk compatibility.
func getCacheControl(providerOpts map[string]any) *anthropicCacheControl {
	if len(providerOpts) == 0 {
		return nil
	}
	anthropicOpts, ok := providerOpts["anthropic"].(map[string]any)
	if !ok {
		return nil
	}
	cc, ok := anthropicOpts["cacheControl"]
	if !ok {
		cc, ok = anthropicOpts["cache_control"]
	}
	if !ok {
		return nil
	}
	ccMap, ok := cc.(map[string]any)
	if !ok {
		return nil
	}
	result := &anthropicCacheControl{Type: "ephemeral"}
	if t, ok := ccMap["type"].(string); ok {
		result.Type = t
	}
	if ttl, ok := ccMap["ttl"].(string); ok {
		result.TTL = ttl
	}
	return result
}
