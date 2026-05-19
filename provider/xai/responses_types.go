package xai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// xAI Responses API request / response types.
//
// These mirror the /v1/responses wire format used by xAI (which inherits
// from OpenAI's Responses API, but with its own finish-reason vocabulary
// and usage-details shape). See:
//
//	references/ai-sdk/packages/xai/src/responses/xai-responses-api.ts
//	references/ai-sdk/packages/xai/src/responses/convert-to-xai-responses-input.ts

type responsesRequest struct {
	Model              string               `json:"model"`
	Input              []responsesInputItem `json:"input"`
	Stream             bool                 `json:"stream,omitempty"`
	Temperature        *float64             `json:"temperature,omitempty"`
	TopP               *float64             `json:"top_p,omitempty"`
	MaxOutputTokens    *int                 `json:"max_output_tokens,omitempty"`
	Tools              []responsesToolWire  `json:"tools,omitempty"`
	ToolChoice         any                  `json:"tool_choice,omitempty"`
	Store              *bool                `json:"store,omitempty"`
	Reasoning          *reasoningConfig     `json:"reasoning,omitempty"`
	Logprobs           *bool                `json:"logprobs,omitempty"`
	TopLogprobs        *int                 `json:"top_logprobs,omitempty"`
	Include            []string             `json:"include,omitempty"`
	PreviousResponseID string               `json:"previous_response_id,omitempty"`
	Text               *textConfig          `json:"text,omitempty"`
}

type reasoningConfig struct {
	Effort string `json:"effort,omitempty"`
}

type textConfig struct {
	Format *textFormat `json:"format,omitempty"`
}

type textFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// responsesInputItem represents a single entry in the "input" array.
// A custom marshaller emits only the fields relevant to the item variant.
type responsesInputItem struct {
	Type string `json:"type,omitempty"`
	Role string `json:"role,omitempty"`

	// system/user/assistant message content (plain text)
	Content string `json:"content,omitempty"`

	// user message content parts (input_text / input_image)
	ContentParts []responsesContentPart `json:"-"`

	// assistant text id
	ID string `json:"id,omitempty"`

	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// function_call_output fields
	Output any `json:"output,omitempty"`

	// reasoning fields
	EncryptedContent string                 `json:"encrypted_content,omitempty"`
	Summary          []responsesSummaryPart `json:"-"`
}

// responsesSummaryPart is a single "summary_text" block on a reasoning item.
type responsesSummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesContentPart is a single element inside a user message's
// content array (input_text / input_image / input_file).
type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	// FileURL carries non-image documents (PDF, text, CSV, …) per
	// https://docs.x.ai/docs/guides/chat-with-files. ai-sdk #14805.
	FileURL string `json:"file_url,omitempty"`
	// FileID references a file uploaded via the xAI Files API.
	FileID string `json:"file_id,omitempty"`
}

// responsesToolWire is a marker interface for wire-format structs that
// serialize into the Responses API `tools` array. Function tools and
// each provider-defined hosted tool implement it so they share one
// []responsesToolWire without a fat union struct.
type responsesToolWire interface {
	isResponsesToolWire()
}

// responsesTool is a function tool declaration in the request body.
type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

func (responsesTool) isResponsesToolWire()            {}
func (xaiHostedWebSearch) isResponsesToolWire()       {}
func (xaiHostedXSearch) isResponsesToolWire()         {}
func (xaiHostedCodeInterpreter) isResponsesToolWire() {}
func (xaiHostedViewImage) isResponsesToolWire()       {}
func (xaiHostedViewXVideo) isResponsesToolWire()      {}
func (xaiHostedFileSearch) isResponsesToolWire()      {}
func (xaiHostedMCP) isResponsesToolWire()             {}

// MarshalJSON emits the right shape per item variant. Mirrors the OpenAI
// Responses converter but without systemMessageMode or phase handling,
// since xAI uses literal "system" and has no codex commentary phase.
func (r responsesInputItem) MarshalJSON() ([]byte, error) {
	// system / user message without multipart content
	if r.Role == "system" {
		return json.Marshal(map[string]any{
			"role":    r.Role,
			"content": r.Content,
		})
	}

	if r.Role == "user" {
		return json.Marshal(map[string]any{
			"role":    r.Role,
			"content": r.ContentParts,
		})
	}

	if r.Role == "assistant" {
		m := map[string]any{
			"role":    r.Role,
			"content": r.Content,
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		return json.Marshal(m)
	}

	if r.Type == "function_call" {
		m := map[string]any{
			"type":      r.Type,
			"call_id":   r.CallID,
			"name":      r.Name,
			"arguments": r.Arguments,
			"status":    "completed",
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		return json.Marshal(m)
	}

	if r.Type == "function_call_output" {
		return json.Marshal(map[string]any{
			"type":    r.Type,
			"call_id": r.CallID,
			"output":  r.Output,
		})
	}

	if r.Type == "reasoning" {
		m := map[string]any{
			"type":    r.Type,
			"summary": r.Summary,
			"status":  "completed",
		}
		if r.ID != "" {
			m["id"] = r.ID
		}
		if r.EncryptedContent != "" {
			m["encrypted_content"] = r.EncryptedContent
		}
		return json.Marshal(m)
	}

	return json.Marshal(map[string]any{
		"type":    r.Type,
		"role":    r.Role,
		"content": r.Content,
	})
}

// Streaming chunk types.

type responsesStreamChunk struct {
	Type     string         `json:"type"`
	Response *responsesData `json:"response,omitempty"`

	OutputIndex int            `json:"output_index,omitempty"`
	Item        *responsesItem `json:"item,omitempty"`

	ItemID string `json:"item_id,omitempty"`
	Delta  string `json:"delta,omitempty"`

	// Annotations are emitted on response.output_text.done; Annotation
	// is emitted on response.output_text.annotation.added (one at a
	// time). Both carry url_citation entries from web_search / x_search.
	Annotations []responsesAnnotation `json:"annotations,omitempty"`
	Annotation  *responsesAnnotation  `json:"annotation,omitempty"`

	Error *responsesError `json:"error,omitempty"`
}

// responsesAnnotation matches xAI's url_citation annotation shape on
// the Responses API. Mirrors ai-sdk's annotationSchema in
// packages/xai/src/responses/xai-responses-api.ts.
type responsesAnnotation struct {
	Type  string `json:"type"`
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`
}

type responsesData struct {
	ID                string          `json:"id"`
	CreatedAt         int64           `json:"created_at,omitempty"`
	Model             string          `json:"model,omitempty"`
	Status            string          `json:"status,omitempty"`
	Usage             *responsesUsage `json:"usage,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
}

type responsesItem struct {
	Type   string `json:"type"`
	ID     string `json:"id,omitempty"`
	Role   string `json:"role,omitempty"`
	Status string `json:"status,omitempty"`

	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	EncryptedContent string `json:"encrypted_content,omitempty"`
}

type responsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens,omitempty"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
}

type responsesError struct {
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
}

// convertToResponsesInput translates goai messages to the xAI Responses
// "input" array. Mirrors ai-sdk's convertToXaiResponsesInput without the
// systemMessageMode branch (xAI always uses the literal "system" role)
// and without codex phase handling.
func convertToResponsesInput(messages []message.Message) []responsesInputItem {
	result := make([]responsesInputItem, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			result = append(result, responsesInputItem{
				Role:    "system",
				Content: getTextFromContent(msg.Content),
			})

		case message.RoleUser:
			result = append(result, responsesInputItem{
				Role:         "user",
				ContentParts: convertToResponsesContentParts(msg.Content),
			})

		case message.RoleAssistant:
			reasoningMessageIndices := make(map[string]int)

			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.TextPart:
					if p.Text != "" {
						var id string
						if xaiOpts, ok := p.ProviderOptions["xai"].(map[string]any); ok {
							if v, ok := xaiOpts["itemId"].(string); ok {
								id = v
							}
						}
						result = append(result, responsesInputItem{
							Role:    "assistant",
							Content: p.Text,
							ID:      id,
						})
					}

				case message.ReasoningPart:
					var reasoningID string
					var encryptedContent string
					if p.ProviderOptions != nil {
						if xaiOpts, ok := p.ProviderOptions["xai"].(map[string]any); ok {
							if v, ok := xaiOpts["itemId"].(string); ok {
								reasoningID = v
							}
							if v, ok := xaiOpts["reasoningEncryptedContent"].(string); ok {
								encryptedContent = v
							}
						}
						// ai-sdk also tolerates flat itemId / reasoningEncryptedContent for xai.
						if reasoningID == "" {
							if v, ok := p.ProviderOptions["itemId"].(string); ok {
								reasoningID = v
							}
						}
						if encryptedContent == "" {
							if v, ok := p.ProviderOptions["reasoningEncryptedContent"].(string); ok {
								encryptedContent = v
							}
						}
					}

					if reasoningID == "" && encryptedContent == "" {
						continue
					}

					if idx, exists := reasoningMessageIndices[reasoningID]; exists && reasoningID != "" {
						if p.Text != "" {
							result[idx].Summary = append(result[idx].Summary, responsesSummaryPart{
								Type: "summary_text",
								Text: p.Text,
							})
						}
						if encryptedContent != "" {
							result[idx].EncryptedContent = encryptedContent
						}
						continue
					}

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
					if reasoningID != "" {
						reasoningMessageIndices[reasoningID] = len(result)
					}
					result = append(result, item)

				case message.ToolCallPart:
					args := string(p.Input)
					if args == "" {
						args = "{}"
					}
					var id string
					if p.ProviderOptions != nil {
						if xaiOpts, ok := p.ProviderOptions["xai"].(map[string]any); ok {
							if v, ok := xaiOpts["itemId"].(string); ok {
								id = v
							}
						}
					}
					result = append(result, responsesInputItem{
						Type:      "function_call",
						ID:        id,
						CallID:    p.ID,
						Name:      p.Name,
						Arguments: args,
					})
				}
			}

			// Fallback for simple text content without Parts
			if len(msg.Content.Parts) == 0 && msg.Content.Text != "" {
				result = append(result, responsesInputItem{
					Role:    "assistant",
					Content: msg.Content.Text,
				})
			}

		case message.RoleTool:
			for _, part := range msg.Content.Parts {
				tr, ok := part.(message.ToolResultPart)
				if !ok {
					continue
				}
				result = append(result, responsesInputItem{
					Type:   "function_call_output",
					CallID: tr.ToolCallID,
					Output: message.ToolOutputWire(tr.Output),
				})
			}
		}
	}

	return result
}

func convertToResponsesContentParts(content message.Content) []responsesContentPart {
	if content.Text != "" && len(content.Parts) == 0 {
		return []responsesContentPart{{Type: "input_text", Text: content.Text}}
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
			if !strings.HasPrefix(imageURL, "http") && !strings.HasPrefix(imageURL, "data:") {
				mime := p.MimeType
				if mime == "" {
					mime = "image/jpeg"
				}
				imageURL = "data:" + mime + ";base64," + imageURL
			}
			result = append(result, responsesContentPart{
				Type:     "input_image",
				ImageURL: imageURL,
			})
		case message.FilePart:
			// xAI Responses API: non-image documents (PDF, text, CSV, …)
			// are supported only via URL or a Files-API reference; inline
			// bytes for non-image files are rejected by the upstream API.
			// ai-sdk #14805. URL-bearing FileParts emit `input_file`;
			// inline-bytes non-image parts are dropped silently (no
			// downstream error path from this helper today — same
			// behavior as before this commit, which had no FilePart case
			// at all).
			if p.URL != "" {
				result = append(result, responsesContentPart{
					Type:    "input_file",
					FileURL: p.URL,
				})
			}
		}
	}
	return result
}

// normalizeResponsesToolChoice adapts a caller-supplied tool choice to
// the xAI Responses API's accepted shapes. The xAI API supports only
// "auto" / "none" / "required" / {"type": "function", "name": "..."};
// it does NOT support forcing a server-side hosted tool (ai-sdk #05f3f36).
// If the caller asks to force a hosted tool by name, drop the forced
// choice (fall back to "auto" by omission). Function-tool forces are
// translated from the V3-shaped {type: "tool", toolName: "X"} to the
// xAI wire shape.
func normalizeResponsesToolChoice(choice any, tools []tool.Tool) any {
	out, _ := normalizeResponsesToolChoiceWithWarnings(choice, tools)
	return out
}

// normalizeResponsesToolChoiceWithWarnings emits an unsupported warning
// when a forced toolChoice targets an xAI hosted tool.
func normalizeResponsesToolChoiceWithWarnings(choice any, tools []tool.Tool) (any, []stream.Warning) {
	m, ok := choice.(map[string]any)
	if !ok {
		// Strings ("auto", "none", "required") or nil pass through.
		return choice, nil
	}
	t, _ := m["type"].(string)
	if t != "tool" {
		return choice, nil
	}
	name, _ := m["toolName"].(string)
	if name == "" {
		return choice, nil
	}
	for _, tl := range tools {
		if tl.Name != name && tl.ProviderID != name {
			continue
		}
		if tl.Type == "provider" && isXaiHostedToolID(tl.ProviderID) {
			// xAI rejects a forced server-side hosted tool.
			warning := stream.UnsupportedWarning(
				"toolChoice",
				fmt.Sprintf("xAI does not support forcing a hosted provider tool %q", tl.ProviderID),
			)
			return nil, []stream.Warning{warning}
		}
		break
	}
	return map[string]any{"type": "function", "name": name}, nil
}

func convertToResponsesTools(tools []tool.Tool) []responsesToolWire {
	out, _ := convertToResponsesToolsWithWarnings(tools)
	return out
}

// convertToResponsesToolsWithWarnings is the warnings-aware variant.
func convertToResponsesToolsWithWarnings(tools []tool.Tool) ([]responsesToolWire, []stream.Warning) {
	result := make([]responsesToolWire, 0, len(tools))
	var warnings []stream.Warning
	for _, t := range tools {
		if t.Type == "provider" {
			hosted, ok := convertXaiProviderTool(t)
			if !ok {
				warnings = append(warnings, stream.UnsupportedWarning(
					"tool",
					fmt.Sprintf("provider-defined tool %q is not supported by xAI Responses", t.ProviderID),
				))
				continue
			}
			result = append(result, hosted)
			continue
		}
		result = append(result, responsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}
	return result, warnings
}

// getTextFromContent returns the plain-text representation of a message
// body, concatenating any TextParts if Parts is populated.
func getTextFromContent(content message.Content) string {
	if content.Text != "" {
		return content.Text
	}
	var b strings.Builder
	for _, part := range content.Parts {
		if tp, ok := part.(message.TextPart); ok {
			b.WriteString(tp.Text)
		}
	}
	return b.String()
}
