package openai

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Responses API request types
// See: https://platform.openai.com/docs/api-reference/responses

type responsesRequest struct {
	Model             string               `json:"model"`
	Input             []responsesInputItem `json:"input"`
	Stream            bool                 `json:"stream,omitempty"`
	Temperature       *float64             `json:"temperature,omitempty"`
	TopP              *float64             `json:"top_p,omitempty"`
	MaxOutputTokens   *int                 `json:"max_output_tokens,omitempty"`
	Tools             []responsesToolWire  `json:"tools,omitempty"`
	ToolChoice        any                  `json:"tool_choice,omitempty"` // "auto", "none", "required", or {type: "function", name: "..."}
	Instructions      string               `json:"instructions,omitempty"`
	Store             *bool                `json:"store,omitempty"`
	Reasoning         *reasoningConfig     `json:"reasoning,omitempty"`
	PromptCacheKey    string               `json:"prompt_cache_key,omitempty"` // Session ID for prompt caching
	Include           []string             `json:"include,omitempty"`          // Extra fields to include: "reasoning.encrypted_content", "file_search_call.results", "message.output_text.logprobs"
	User              string               `json:"user,omitempty"`             // Unique identifier for end-user
	ParallelToolCalls *bool                `json:"parallel_tool_calls,omitempty"`
	Metadata          any                  `json:"metadata,omitempty"`
	Text              *textConfig          `json:"text,omitempty"`                   // Text verbosity config
	Truncation        string               `json:"truncation,omitempty"`             // "auto" or "disabled"
	ServiceTier       string               `json:"service_tier,omitempty"`           // "auto", "flex", "priority", "default"
	SafetyIdentifier  string               `json:"safety_identifier,omitempty"`      // Safety monitoring identifier
	PromptCacheRetent string               `json:"prompt_cache_retention,omitempty"` // "in_memory" or "24h"
}

// textConfig controls text output behavior
type textConfig struct {
	Format    *textFormat `json:"format,omitempty"`
	Verbosity string      `json:"verbosity,omitempty"` // "low", "medium", "high"
}

// textFormat configures structured output (json_object or json_schema).
// Mirrors ai-sdk's openai-responses-language-model.ts response format.
type textFormat struct {
	Type        string          `json:"type"`                  // "json_object" or "json_schema"
	Name        string          `json:"name,omitempty"`        // schema name (json_schema)
	Description string          `json:"description,omitempty"` // schema description (json_schema)
	Schema      json.RawMessage `json:"schema,omitempty"`      // JSON schema (json_schema)
	Strict      *bool           `json:"strict,omitempty"`      // strict schema validation (json_schema)
}

// reasoningConfig controls reasoning/thinking behavior for supported models
type reasoningConfig struct {
	Effort  string `json:"effort,omitempty"`  // "none", "minimal", "low", "medium", "high", "xhigh"
	Summary string `json:"summary,omitempty"` // "auto", "detailed" - controls reasoning summary output
}

// responsesInputItem represents an item in the input array
type responsesInputItem struct {
	// Common fields
	Type string `json:"type,omitempty"`
	Role string `json:"role,omitempty"`

	// For system/developer messages
	Content string `json:"content,omitempty"`

	// For user messages (array of content parts)
	ContentParts []responsesContentPart `json:"-"`

	// For assistant messages
	AssistantContent []responsesOutputText `json:"-"`
	ID               string                `json:"id,omitempty"`
	// Phase is "commentary" | "final_answer" (gpt-5.3-codex+, ai-sdk
	// #66a374c). Emitted on the assistant message when the caller
	// forwards a prior response's phase via providerOptions.openai.phase.
	Phase string `json:"-"`

	// For function_call
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// For function_call_output — string for text, []responsesContentPart for multipart
	Output any `json:"output,omitempty"`

	// For reasoning
	EncryptedContent string                 `json:"encrypted_content,omitempty"`
	Summary          []responsesSummaryPart `json:"-"`
}

// responsesSummaryPart represents a summary text part in reasoning
type responsesSummaryPart struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// MarshalJSON handles the custom serialization for responsesInputItem
func (r responsesInputItem) MarshalJSON() ([]byte, error) {
	// For system/developer messages
	if r.Role == "system" || r.Role == "developer" {
		return json.Marshal(map[string]any{
			"role":    r.Role,
			"content": r.Content,
		})
	}

	// For user messages
	if r.Role == "user" {
		return json.Marshal(map[string]any{
			"role":    r.Role,
			"content": r.ContentParts,
		})
	}

	// For assistant messages
	if r.Role == "assistant" {
		m := map[string]any{
			"role":    r.Role,
			"content": r.AssistantContent,
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		// Echo back phase on follow-up requests (ai-sdk #66a374c).
		if r.Phase != "" {
			m["phase"] = r.Phase
		}
		return json.Marshal(m)
	}

	// For function_call
	if r.Type == "function_call" {
		m := map[string]any{
			"type":      r.Type,
			"call_id":   r.CallID,
			"name":      r.Name,
			"arguments": r.Arguments,
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		return json.Marshal(m)
	}

	// For function_call_output
	if r.Type == "function_call_output" {
		return json.Marshal(map[string]any{
			"type":    r.Type,
			"call_id": r.CallID,
			"output":  r.Output,
		})
	}

	// For reasoning
	if r.Type == "reasoning" {
		m := map[string]any{
			"type":    r.Type,
			"summary": r.Summary,
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		if r.EncryptedContent != "" {
			m["encrypted_content"] = r.EncryptedContent
		}
		return json.Marshal(m)
	}

	// Default: marshal as-is
	return json.Marshal(map[string]any{
		"type":    r.Type,
		"role":    r.Role,
		"content": r.Content,
	})
}

type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
}

type responsesOutputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesToolWire is a marker interface for wire-format structs that
// serialize into the Responses API `tools` array. Function tools and
// each provider-defined hosted tool implement it so they share one
// []responsesToolWire without a fat union struct.
type responsesToolWire interface {
	isResponsesToolWire()
}

type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"` // Pointer to allow explicit false
}

func (responsesTool) isResponsesToolWire()          {}
func (openaiHostedWebSearch) isResponsesToolWire()  {}
func (openaiHostedCustom) isResponsesToolWire()     {}
func (openaiHostedToolSearch) isResponsesToolWire() {}

// Responses API streaming chunk types

type responsesChunk struct {
	Type     string         `json:"type"`
	Response *responsesData `json:"response,omitempty"`

	// For output_item events
	OutputIndex int            `json:"output_index,omitempty"`
	Item        *responsesItem `json:"item,omitempty"`

	// For delta events
	ItemID string `json:"item_id,omitempty"`
	Delta  string `json:"delta,omitempty"`

	// For annotation events
	Annotation *responsesAnnotation `json:"annotation,omitempty"`

	// For error events
	Error *responsesError `json:"error,omitempty"`
}

type responsesData struct {
	ID                string          `json:"id"`
	CreatedAt         int64           `json:"created_at"`
	Model             string          `json:"model"`
	Usage             *responsesUsage `json:"usage,omitempty"`
	ServiceTier       string          `json:"service_tier,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
}

type responsesItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Role   string `json:"role,omitempty"`
	Status string `json:"status,omitempty"`

	// For function_call
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// For reasoning
	EncryptedContent string `json:"encrypted_content,omitempty"`

	// For assistant message items (gpt-5.3-codex+, ai-sdk #66a374c).
	// Values: "commentary" or "final_answer". Surfaces via
	// providerMetadata.openai.phase on text-start/end events.
	Phase string `json:"phase,omitempty"`
}

type responsesAnnotation struct {
	Type       string `json:"type"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
}

type responsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
}

type responsesError struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
}

// Conversion functions for Responses API

// SystemMessageMode controls how system messages are mapped in the Responses API.
// Options: "system", "developer", "remove"
// - "system": Use the 'system' role (default for most models)
// - "developer": Use the 'developer' role (used by reasoning models like o1, o3)
// - "remove": Remove system messages (for models that don't support them)
type SystemMessageMode string

const (
	SystemMessageModeSystem    SystemMessageMode = "system"
	SystemMessageModeDeveloper SystemMessageMode = "developer"
	SystemMessageModeRemove    SystemMessageMode = "remove"
)

// ConversionWarning represents a warning generated during message conversion.
// Matches ai-sdk's SharedV3Warning type.
type ConversionWarning struct {
	Type    string `json:"type"`    // "other"
	Message string `json:"message"` // Warning message
}

// ConversionResult contains the converted input and any warnings.
type ConversionResult struct {
	Input    []responsesInputItem
	Warnings []ConversionWarning
}

func convertToResponsesInput(messages []message.Message, systemMessageMode string) []responsesInputItem {
	result := convertToResponsesInputWithWarnings(messages, systemMessageMode)
	return result.Input
}

// convertToResponsesInputWithWarnings converts messages to Responses API input format
// and returns any warnings generated during conversion.
// This matches ai-sdk's convertToOpenAIResponsesInput behavior.
func convertToResponsesInputWithWarnings(messages []message.Message, systemMessageMode string) ConversionResult {
	result := make([]responsesInputItem, 0, len(messages))
	var warnings []ConversionWarning

	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			// Map system messages based on systemMessageMode setting
			switch systemMessageMode {
			case "system":
				result = append(result, responsesInputItem{
					Role:    "system",
					Content: getTextFromContent(msg.Content),
				})
			case "developer":
				result = append(result, responsesInputItem{
					Role:    "developer",
					Content: getTextFromContent(msg.Content),
				})
			case "remove":
				// Skip system messages
				warnings = append(warnings, ConversionWarning{
					Type:    "other",
					Message: "system messages are removed for this model",
				})
				continue
			default:
				// Default to developer (matching ai-sdk behavior for newer models)
				result = append(result, responsesInputItem{
					Role:    "developer",
					Content: getTextFromContent(msg.Content),
				})
			}
		case message.RoleUser:
			item := responsesInputItem{
				Role: "user",
			}
			item.ContentParts = convertToResponsesContentParts(msg.Content)
			result = append(result, item)
		case message.RoleAssistant:
			// Process parts in order (matching ai-sdk behavior)
			// ai-sdk iterates through parts in order, not grouped by type
			reasoningMessageIndices := make(map[string]int)

			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.TextPart:
					// Add text as assistant message. Forward a prior
					// response's `phase` (ai-sdk #66a374c) when provided
					// via providerOptions.openai.phase on the text part —
					// required for gpt-5.3-codex continuity.
					if p.Text != "" {
						var phase string
						if openaiOpts, ok := p.ProviderOptions["openai"].(map[string]any); ok {
							if ph, ok := openaiOpts["phase"].(string); ok {
								phase = ph
							}
						}
						result = append(result, responsesInputItem{
							Role: "assistant",
							AssistantContent: []responsesOutputText{
								{Type: "output_text", Text: p.Text},
							},
							Phase: phase,
						})
					}
				case message.ReasoningPart:
					// Get itemId from provider options
					var reasoningID string
					var encryptedContent string
					if p.ProviderOptions != nil {
						if id, ok := p.ProviderOptions["itemId"].(string); ok {
							reasoningID = id
						}
						if ec, ok := p.ProviderOptions["reasoningEncryptedContent"].(string); ok {
							encryptedContent = ec
						}
					}

					// ai-sdk #5e18272: a reasoning part without itemId is
					// still valid as long as encrypted_content is present.
					// The OpenAI Responses API accepts reasoning items
					// without an id when encrypted_content is provided,
					// which enables multi-turn reasoning round-trip in
					// store:false / ZDR mode where server-side item
					// persistence isn't used.
					if reasoningID == "" {
						if encryptedContent == "" {
							warnings = append(warnings, ConversionWarning{
								Type:    "other",
								Message: "Non-OpenAI reasoning parts are not supported. Skipping reasoning part.",
							})
							continue
						}
						item := responsesInputItem{
							Type:             "reasoning",
							EncryptedContent: encryptedContent,
							Summary:          []responsesSummaryPart{},
						}
						if p.Text != "" {
							item.Summary = append(item.Summary, responsesSummaryPart{
								Type: "summary_text",
								Text: p.Text,
							})
						}
						result = append(result, item)
						continue
					}

					// Check if we already have a reasoning message with this ID
					if idx, exists := reasoningMessageIndices[reasoningID]; exists {
						// Merge: add summary text to existing message in result slice
						if p.Text != "" {
							result[idx].Summary = append(result[idx].Summary, responsesSummaryPart{
								Type: "summary_text",
								Text: p.Text,
							})
						} else {
							// Warn about empty reasoning parts being appended
							warnings = append(warnings, ConversionWarning{
								Type:    "other",
								Message: "Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part.",
							})
						}
						// Update encrypted content if provided (can be set in last part)
						if encryptedContent != "" {
							result[idx].EncryptedContent = encryptedContent
						}
					} else {
						// Create new reasoning message
						item := responsesInputItem{
							Type:             "reasoning",
							ID:               reasoningID,
							EncryptedContent: encryptedContent,
							Summary:          []responsesSummaryPart{},
						}
						if p.Text != "" {
							item.Summary = append(item.Summary, responsesSummaryPart{
								Type: "summary_text",
								Text: p.Text,
							})
						}
						reasoningMessageIndices[reasoningID] = len(result)
						result = append(result, item)
					}
				case message.ToolCallPart:
					// Add tool call as function_call item.
					// Default undefined/empty input to "{}" so the Responses
					// API never sees a malformed arguments string
					// (ai-sdk #953385d).
					args := string(p.Input)
					if args == "" {
						args = "{}"
					}
					result = append(result, responsesInputItem{
						Type:      "function_call",
						CallID:    p.ID,
						Name:      p.Name,
						Arguments: args,
					})
				}
			}

			// Handle simple text content if no parts (fallback for Content.Text without Parts)
			if len(msg.Content.Parts) == 0 && msg.Content.Text != "" {
				result = append(result, responsesInputItem{
					Role: "assistant",
					AssistantContent: []responsesOutputText{
						{Type: "output_text", Text: msg.Content.Text},
					},
				})
			}
		case message.RoleTool:
			// ai-sdk puts image/file attachments inside function_call_output.output
			// as an array of content parts. Collect all parts first, then build outputs.
			type toolOutput struct {
				callID string
				text   string
			}
			var toolOutputs []toolOutput
			var attachments []responsesContentPart

			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.ToolResultPart:
					textOutput := ""
					switch v := p.Result.(type) {
					case string:
						textOutput = v
					default:
						if b, err := json.Marshal(v); err == nil {
							textOutput = string(b)
						}
					}
					toolOutputs = append(toolOutputs, toolOutput{callID: p.ToolCallID, text: textOutput})
				case message.ImagePart:
					imageURL := p.Image
					if !strings.HasPrefix(imageURL, "http") && !strings.HasPrefix(imageURL, "data:") {
						mime := p.MimeType
						if mime == "" {
							mime = "image/png"
						}
						imageURL = "data:" + mime + ";base64," + imageURL
					}
					attachments = append(attachments, responsesContentPart{
						Type:     "input_image",
						ImageURL: imageURL,
					})
				case message.FilePart:
					// ai-sdk #bc01093: file-url in tool output. A FilePart
					// with URL becomes an input_file with file_url; base64
					// PDFs still emit file_data.
					if p.URL != "" {
						attachments = append(attachments, responsesContentPart{
							Type:     "input_file",
							Filename: p.Filename,
							FileURL:  p.URL,
						})
					} else if p.MimeType == "application/pdf" {
						filename := p.Filename
						if filename == "" {
							filename = "document.pdf"
						}
						attachments = append(attachments, responsesContentPart{
							Type:     "input_file",
							Filename: filename,
							FileData: "data:application/pdf;base64," + p.Data,
						})
					}
				}
			}

			for i, to := range toolOutputs {
				var output any
				// Attach images/files to the last tool output (matches ai-sdk behavior).
				if i == len(toolOutputs)-1 && len(attachments) > 0 {
					parts := make([]responsesContentPart, 0, len(attachments)+1)
					if to.text != "" {
						parts = append(parts, responsesContentPart{Type: "input_text", Text: to.text})
					}
					parts = append(parts, attachments...)
					output = parts
				} else {
					output = to.text
				}
				result = append(result, responsesInputItem{
					Type:   "function_call_output",
					CallID: to.callID,
					Output: output,
				})
			}
		}
	}

	return ConversionResult{
		Input:    result,
		Warnings: warnings,
	}
}

func convertToResponsesContentParts(content message.Content) []responsesContentPart {
	// If simple text, return single text part
	if content.Text != "" && len(content.Parts) == 0 {
		return []responsesContentPart{
			{Type: "input_text", Text: content.Text},
		}
	}

	result := make([]responsesContentPart, 0, len(content.Parts))
	for _, part := range content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			result = append(result, responsesContentPart{
				Type: "input_text",
				Text: p.Text,
			})
		case message.ImagePart:
			imageURL := p.Image
			// If not already a URL or data URI, wrap base64 data in a data URI.
			if !strings.HasPrefix(imageURL, "http") && !strings.HasPrefix(imageURL, "data:") {
				mime := p.MimeType
				if mime == "" {
					mime = "image/png"
				}
				imageURL = "data:" + mime + ";base64," + imageURL
			}
			result = append(result, responsesContentPart{
				Type:     "input_image",
				ImageURL: imageURL,
			})
		case message.FilePart:
			// A URL-bearing FilePart maps to input_file with file_url
			// regardless of mediaType (#bc01093). Base64 file parts are
			// still restricted to PDF since that's all OpenAI Responses
			// accepts as file_data.
			if p.URL != "" {
				result = append(result, responsesContentPart{
					Type:     "input_file",
					Filename: p.Filename,
					FileURL:  p.URL,
				})
			} else if p.MimeType == "application/pdf" {
				filename := p.Filename
				if filename == "" {
					filename = "document.pdf"
				}
				result = append(result, responsesContentPart{
					Type:     "input_file",
					Filename: filename,
					FileData: "data:application/pdf;base64," + p.Data,
				})
			}
		}
	}
	return result
}

func convertToResponsesTools(tools []tool.Tool, strictJsonSchema bool) []responsesToolWire {
	result, _ := convertToResponsesToolsWithWarnings(tools, strictJsonSchema)
	return result
}

// convertToResponsesToolsWithWarnings is the warnings-aware variant. Caller
// receives an Unsupported warning for each unknown provider-defined tool so
// the drop is visible in the stream surface.
func convertToResponsesToolsWithWarnings(tools []tool.Tool, strictJsonSchema bool) ([]responsesToolWire, []stream.Warning) {
	result := make([]responsesToolWire, 0, len(tools))
	var warnings []stream.Warning

	for _, t := range tools {
		if t.Type == "provider" {
			hosted, ok := convertOpenAIProviderTool(t)
			if !ok {
				warnings = append(warnings, stream.UnsupportedWarning(
					"tool",
					fmt.Sprintf("provider-defined tool %q is not supported by OpenAI Responses", t.ProviderID),
				))
				continue
			}
			result = append(result, hosted)
			continue
		}

		var schema json.RawMessage
		if strictJsonSchema {
			// Ensure schema has additionalProperties: false for strict mode
			schema = ensureStrictSchema(t.InputSchema)
		} else {
			schema = t.InputSchema
		}

		strict := strictJsonSchema
		result = append(result, responsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schema,
			Strict:      &strict,
		})
	}

	return result, warnings
}

// ensureStrictSchema modifies a JSON schema to be compatible with OpenAI's strict mode:
// 1. Adds "additionalProperties": false to object types
// 2. Ensures all properties are listed in "required"
// This is required for OpenAI's Responses API with strict mode.
func ensureStrictSchema(schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 {
		return schema
	}

	var obj map[string]any
	if err := json.Unmarshal(schema, &obj); err != nil {
		return schema
	}

	// Recursively process the schema
	processSchemaObject(obj)

	result, err := json.Marshal(obj)
	if err != nil {
		return schema
	}
	return result
}

// processSchemaObject recursively processes a schema object to ensure strict mode compliance
func processSchemaObject(obj map[string]any) {
	schemaType, _ := obj["type"].(string)

	// For object types, add additionalProperties: false and ensure all properties in required
	if schemaType == "object" {
		if _, hasAdditionalProps := obj["additionalProperties"]; !hasAdditionalProps {
			obj["additionalProperties"] = false
		}

		// Ensure all properties are in required (preserving existing order, adding missing ones)
		if props, ok := obj["properties"].(map[string]any); ok {
			// Get existing required array
			existingRequired := make([]string, 0)
			if req, ok := obj["required"].([]any); ok {
				for _, r := range req {
					if s, ok := r.(string); ok {
						existingRequired = append(existingRequired, s)
					}
				}
			} else if req, ok := obj["required"].([]string); ok {
				existingRequired = req
			}

			// Build set of existing required properties
			requiredSet := make(map[string]bool)
			for _, r := range existingRequired {
				requiredSet[r] = true
			}

			// Add any missing properties to required (sorted, so they're added deterministically)
			missingKeys := make([]string, 0)
			for key := range props {
				if !requiredSet[key] {
					missingKeys = append(missingKeys, key)
				}
			}
			sort.Strings(missingKeys) // Only sort the NEW keys being added

			// Combine: existing required + sorted missing keys
			allRequired := append(existingRequired, missingKeys...)
			obj["required"] = allRequired

			// Recursively process each property
			for key, prop := range props {
				if propMap, ok := prop.(map[string]any); ok {
					processSchemaObject(propMap)
					props[key] = propMap
				}
			}
			obj["properties"] = props
		}
	}

	// For array types, process the items schema
	if schemaType == "array" {
		if items, ok := obj["items"].(map[string]any); ok {
			processSchemaObject(items)
			obj["items"] = items
		}
	}
}

func mapResponsesFinishReason(reason string, hasFunctionCall bool) stream.FinishReason {
	// If there's an incomplete_details reason
	switch reason {
	case "max_output_tokens":
		return stream.FinishReasonLength
	case "content_filter":
		return stream.FinishReasonContentFilter
	case "":
		// No incomplete reason - check if we had function calls
		if hasFunctionCall {
			return stream.FinishReasonToolCalls
		}
		return stream.FinishReasonStop
	default:
		return stream.FinishReasonOther
	}
}
