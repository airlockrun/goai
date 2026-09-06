// Package openaicompat provides an OpenAI-compatible provider base.
// Many providers (Groq, Together, Fireworks, etc.) use OpenAI-compatible APIs.
package openaicompat

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

const maxErrorResponseBodyBytes = 8 << 10

// RequestModifier allows providers to add extra fields to the request body.
// It receives the provider options from CallOptions.ProviderOptions and returns
// additional fields to merge into the request JSON plus any warnings about
// unsupported or silently-converted options.
type RequestModifier func(providerOptions map[string]any) (extraFields map[string]any, warnings []stream.Warning, err error)

// CallWarner is an optional hook that inspects the full CallOptions and
// emits provider-specific warnings (e.g. "topK is not supported"). Separate
// from RequestModifier because that one only sees providerOptions.
type CallWarner func(options *stream.CallOptions) []stream.Warning

// MessageConverter, when set, replaces the default goai-message → chat-message
// conversion. It receives the model ID and the original goai messages and
// returns a slice of message objects ready for JSON marshaling (each element
// can be a ChatMessage, a map[string]any, or any other JSON-marshalable value).
// Use this when a provider needs model-specific message conversion — for
// example DeepSeek's different reasoning_content rules across deepseek-v4
// (echo) vs deepseek-reasoner (strip). When nil, the default
// ConvertToChatMessages output is used.
type MessageConverter func(modelID string, messages []message.Message) ([]any, error)

// Options contains configuration for an OpenAI-compatible provider.
type Options struct {
	// ProviderID is the unique provider identifier.
	ProviderID string

	// BaseURL is the API base URL.
	BaseURL string

	// APIKey is the API key.
	APIKey string

	// HTTPClient sends streaming requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// Headers are additional HTTP headers to send.
	Headers map[string]string

	// AuthHeader is the authorization header name (default: "Authorization").
	AuthHeader string

	// AuthPrefix is the authorization prefix (default: "Bearer ").
	AuthPrefix string

	// RequestModifier is called during request building to add provider-specific fields.
	// Providers can use this to apply their typed options to the request.
	RequestModifier RequestModifier

	// CallWarner runs against the full CallOptions and collects warnings
	// for unsupported CallOption fields (e.g. topK, frequencyPenalty).
	// Matches the per-provider inventory in ai-sdk's language models.
	CallWarner CallWarner
	// ModelCallWarner reports model-specific unsupported call settings.
	ModelCallWarner func(string, *stream.CallOptions) []stream.Warning

	// MessageConverter, when set, replaces the default message conversion.
	// See the MessageConverter type doc for details.
	MessageConverter MessageConverter

	// SupportsStructuredOutputs indicates the provider's endpoint honors the
	// OpenAI "json_schema" response_format with strict decoding. When false,
	// a schema request falls back to "json_object" plus prompt injection.
	// Mirrors ai-sdk's OpenAICompatibleChatConfig.supportsStructuredOutputs.
	SupportsStructuredOutputs bool
	// DefaultStrictJSONSchema controls the default strict flag for native schemas.
	DefaultStrictJSONSchema *bool

	// IncludeUsage controls whether streaming requests send
	// stream_options.include_usage. Nil preserves the default of true.
	IncludeUsage *bool

	// ToolSchemaTransformer, when set, rewrites each tool's JSON-schema
	// parameters before they reach the request body. Providers whose API
	// rejects standard schema keywords use it to sanitize (e.g. xAI strips
	// additionalProperties:false).
	ToolSchemaTransformer func(json.RawMessage) json.RawMessage

	// TransformRequest applies model-specific wire normalization after option conversion.
	TransformRequest func(modelID string, body map[string]any) error
	// MaxEmbeddingInputs overrides the compatible embedding batch limit.
	MaxEmbeddingInputs int
	SupportsPenalties  bool
}

// Provider implements an OpenAI-compatible provider.
type Provider struct {
	opts Options
}

// New creates a new OpenAI-compatible provider.
func New(opts Options) *Provider {
	opts.BaseURL = strings.TrimRight(opts.BaseURL, "/")
	if opts.AuthHeader == "" {
		opts.AuthHeader = "Authorization"
	}
	if opts.AuthPrefix == "" {
		opts.AuthPrefix = "Bearer "
	}
	return &Provider{opts: opts}
}

// ID returns the provider identifier.
func (p *Provider) ID() string {
	return p.opts.ProviderID
}

// BaseURL returns the base URL.
func (p *Provider) BaseURL() string {
	return p.opts.BaseURL
}

// APIKey returns the API key.
func (p *Provider) APIKey() string {
	return p.opts.APIKey
}

// Model returns a language model instance.
func (p *Provider) Model(modelID string) stream.Model {
	return &CompatModel{
		id:       modelID,
		provider: p,
	}
}

// CompatModel represents an OpenAI-compatible model.
type CompatModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *CompatModel) ID() string {
	return m.id
}

// Provider returns the provider ID.
func (m *CompatModel) Provider() string {
	return m.provider.opts.ProviderID
}

// Stream sends a streaming request using the OpenAI-compatible API.
func (m *CompatModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		return nil, err
	}

	requestURL := m.provider.opts.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", requestURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if m.provider.opts.APIKey != "" {
		req.Header.Set(m.provider.opts.AuthHeader, m.provider.opts.AuthPrefix+m.provider.opts.APIKey)
	}
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	client := m.provider.opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message:           "cannot connect to API: " + err.Error(),
			URL:               requestURL,
			RequestBodyValues: json.RawMessage(reqBody),
			Cause:             err,
			IsRetryable:       true,
			IsRetryableSet:    true,
		})
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorResponseBodyBytes+1))
		truncated := len(body) > maxErrorResponseBodyBytes
		if truncated {
			body = body[:maxErrorResponseBodyBytes]
		}
		bodyText := string(body)
		if truncated {
			bodyText += "... (truncated)"
		}
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message:           fmt.Sprintf("%s API error: %s", m.provider.opts.ProviderID, bodyText),
			URL:               requestURL,
			RequestBodyValues: json.RawMessage(reqBody),
			StatusCode:        resp.StatusCode,
			ResponseHeaders:   flattenHeaders(resp.Header),
			ResponseBody:      bodyText,
		})
	}

	events := make(chan stream.Event, 100)
	go func() {
		defer close(events)
		defer resp.Body.Close()
		events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}
		m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
	}()
	return events, nil
}

func flattenHeaders(headers http.Header) map[string]string {
	out := make(map[string]string, len(headers))
	for name, values := range headers {
		out[name] = strings.Join(values, ", ")
	}
	return out
}

func (m *CompatModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning
	if m.provider.opts.CallWarner != nil {
		warnings = append(warnings, m.provider.opts.CallWarner(options)...)
	}
	if m.provider.opts.ModelCallWarner != nil {
		warnings = append(warnings, m.provider.opts.ModelCallWarner(m.id, options)...)
	}

	// Repair any assistant tool_call left unanswered by a tool message before
	// either conversion path runs — Chat Completions (and DeepSeek in
	// particular) reject an unpaired tool_call with HTTP 400.
	messages := pairToolResults(options.Messages)

	// Map ResponseFormat. When the provider lacks native json_schema support
	// but the caller gave a schema, fall back to json_object + inject the
	// schema into the system prompt so the model still knows the shape.
	var respFormat *responseFormat
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		schema := options.ResponseFormat.Schema
		switch {
		case len(schema) > 0 && m.provider.opts.SupportsStructuredOutputs:
			name := options.ResponseFormat.Name
			if name == "" {
				name = "response"
			}
			strictValue := true
			if m.provider.opts.DefaultStrictJSONSchema != nil {
				strictValue = *m.provider.opts.DefaultStrictJSONSchema
			}
			strict := &strictValue
			if options.ProviderOptions != nil {
				if v, ok := options.ProviderOptions["strictJsonSchema"].(bool); ok {
					strict = &v
				}
			}
			respFormat = &responseFormat{
				Type: "json_schema",
				JSONSchema: &responseJSONSchema{
					Name:        name,
					Description: options.ResponseFormat.Description,
					Schema:      schema,
					Strict:      strict,
				},
			}
		case len(schema) > 0:
			respFormat = &responseFormat{Type: "json_object"}
			messages = provider.InjectJSONInstruction(messages, schema)
		default:
			respFormat = &responseFormat{Type: "json_object"}
		}
	}

	var convertedMessages []any
	if m.provider.opts.MessageConverter != nil {
		custom, err := m.provider.opts.MessageConverter(m.id, messages)
		if err != nil {
			return nil, warnings, fmt.Errorf("message converter error: %w", err)
		}
		convertedMessages = custom
	} else {
		raw := convertToMessages(messages)
		convertedMessages = make([]any, len(raw))
		for i := range raw {
			convertedMessages[i] = raw[i]
		}
	}

	req := chatRequest{
		Model:          m.id,
		Stream:         true,
		Messages:       convertedMessages,
		ResponseFormat: respFormat,
	}
	if options.Reasoning != "" && options.Reasoning != "provider-default" {
		req.ReasoningEffort = options.Reasoning
	}
	if m.provider.opts.SupportsPenalties {
		req.FrequencyPenalty = options.FrequencyPenalty
		req.PresencePenalty = options.PresencePenalty
	}

	if options.Temperature != nil {
		req.Temperature = options.Temperature
	}
	if options.TopP != nil {
		req.TopP = options.TopP
	}
	if options.MaxOutputTokens != nil {
		req.MaxTokens = options.MaxOutputTokens
	}
	if len(options.StopSequences) > 0 {
		req.Stop = options.StopSequences
	}

	// Add tools (already ordered by core)
	if len(options.Tools) > 0 {
		req.Tools = convertToTools(options.Tools, m.provider.opts.ToolSchemaTransformer)
	}

	// Translate goai's loose ToolChoice (bare strings or ai-sdk-shaped objects)
	// into Chat Completions' tool_choice. Most OpenAI-compatible APIs accept
	// the same shape. Mirrors ai-sdk parity:
	// packages/openai/src/chat/openai-chat-prepare-tools.ts.
	if options.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(options.ToolChoice)
	}

	includeUsage := m.provider.opts.IncludeUsage == nil || *m.provider.opts.IncludeUsage
	if includeUsage {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	// Apply provider-specific request modifications
	if m.provider.opts.RequestModifier != nil {
		extraFields, extraWarnings, err := m.provider.opts.RequestModifier(options.ProviderOptions)
		if err != nil {
			return nil, warnings, fmt.Errorf("request modifier error: %w", err)
		}
		warnings = append(warnings, extraWarnings...)
		if len(extraFields) > 0 {
			// Marshal the base request
			baseJSON, err := json.Marshal(req)
			if err != nil {
				return nil, warnings, err
			}
			// Unmarshal into a map
			var reqMap map[string]any
			if err := json.Unmarshal(baseJSON, &reqMap); err != nil {
				return nil, warnings, err
			}
			// Merge extra fields
			for k, v := range extraFields {
				reqMap[k] = v
			}
			if m.provider.opts.TransformRequest != nil {
				if err := m.provider.opts.TransformRequest(m.id, reqMap); err != nil {
					return nil, warnings, err
				}
			}
			body, err := json.Marshal(reqMap)
			return body, warnings, err
		}
	}

	body, err := json.Marshal(req)
	if err == nil && m.provider.opts.TransformRequest != nil {
		var fields map[string]any
		if err := json.Unmarshal(body, &fields); err != nil {
			return nil, warnings, err
		}
		if err := m.provider.opts.TransformRequest(m.id, fields); err != nil {
			return nil, warnings, err
		}
		body, err = json.Marshal(fields)
	}
	return body, warnings, err
}

func (m *CompatModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted, reasoningStarted bool
	var currentToolCalls = make(map[int]*toolCallAccumulator)
	toolIDs := make(map[string]int)
	toolIndexes := make(map[int]int)
	lastToolIndex := -1
	var usage stream.Usage
	var usageRaw map[string]any
	var finishReason stream.FinishReason

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: &goaierrors.JSONParseError{Text: data, Cause: err}}}
			return
		}
		if len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			var detail struct {
				Message string
				Type    string
				Code    any
				Param   string
			}
			if err := json.Unmarshal(chunk.Error, &detail); err != nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: invalid error payload: %s", goaierrors.ErrInvalidResponse, chunk.Error)}}
			} else {
				code := ""
				if detail.Code != nil {
					code = fmt.Sprint(detail.Code)
				}
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: &goaierrors.APIError{Provider: m.Provider(), Message: detail.Message, Code: code, Type: detail.Type, Param: detail.Param}}}
			}
			return
		}
		if chunk.Choices == nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: missing choices", goaierrors.ErrInvalidResponse)}}
			return
		}

		// Handle usage
		if len(chunk.UsageRaw) > 0 && string(chunk.UsageRaw) != "null" {
			var typed chatUsage
			if err := json.Unmarshal(chunk.UsageRaw, &typed); err != nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: &goaierrors.JSONParseError{Text: string(chunk.UsageRaw), Cause: err}}}
				return
			}
			usage = usageFromChat(typed)
			var raw map[string]any
			if err := json.Unmarshal(chunk.UsageRaw, &raw); err == nil {
				usageRaw = raw
				usage.Raw = raw
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Handle finish reason
		if choice.FinishReason != "" {
			finishReason = mapFinishReason(choice.FinishReason)
		}

		delta := choice.Delta

		// OpenAI-compatible reasoning models use reasoning_content (and a few
		// endpoints use reasoning). Preserve it as first-class stream events;
		// DeepSeek V4 requires this content to be echoed on later turns.
		reasoningContent := delta.ReasoningContent
		if reasoningContent == "" {
			reasoningContent = delta.Reasoning
		}
		content := delta.Content
		if reasoningContent != "" {
			content = append(chatContent{{Type: "reasoning", Text: reasoningContent}}, content...)
		}
		for _, part := range content {
			if part.Text == "" {
				continue
			}
			if part.Type == "reasoning" {
				if textStarted {
					events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
					textStarted = false
				}
				if !reasoningStarted {
					reasoningStarted = true
					events <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "reasoning-0"}}
				}
				events <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "reasoning-0", Text: part.Text}}
			}

			// Handle text content
			if part.Type == "text" {
				if reasoningStarted {
					events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning-0"}}
					reasoningStarted = false
				}
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: part.Text}}
			}
		}

		// Handle tool calls. Buffers id + arguments until function.name
		// arrives. Some openai-compatible providers send the first delta
		// without function.name and supply it on a later chunk; emitting
		// tool-input-start before the name would produce malformed events.
		// Mirrors ai-sdk PR #14760.
		if len(delta.ToolCalls) > 0 && reasoningStarted {
			events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning-0"}}
			reasoningStarted = false
		}
		for _, tc := range delta.ToolCalls {
			index := lastToolIndex
			if tc.Index != nil {
				index = *tc.Index
				if known, ok := toolIndexes[*tc.Index]; ok {
					index = known
				}
			}
			if known, ok := toolIDs[tc.ID]; tc.ID != "" && ok {
				index = known
			} else if tc.ID != "" && (tc.Index == nil || (currentToolCalls[index] != nil && currentToolCalls[index].id != "" && currentToolCalls[index].id != tc.ID)) {
				index = len(currentToolCalls)
				for currentToolCalls[index] != nil {
					index++
				}
			}
			if index < 0 {
				index = 0
			}
			lastToolIndex = index
			if tc.Index != nil {
				toolIndexes[*tc.Index] = index
			}
			acc, exists := currentToolCalls[index]
			if !exists {
				acc = &toolCallAccumulator{index: index}
				currentToolCalls[index] = acc
			}
			if tc.ID != "" && acc.id == "" {
				acc.id = tc.ID
				toolIDs[tc.ID] = index
			}

			if !acc.started {
				acc.arguments += tc.Function.Arguments
				if tc.Function.Name != "" {
					acc.name = tc.Function.Name
				}
				if acc.name != "" && acc.id != "" {
					acc.started = true
					events <- stream.Event{
						Type: stream.EventToolInputStart,
						Data: stream.ToolInputStartEvent{ID: acc.id, ToolName: acc.name},
					}
					if acc.arguments != "" {
						events <- stream.Event{
							Type: stream.EventToolInputDelta,
							Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: acc.arguments},
						}
					}
				}
				continue
			}

			if tc.Function.Arguments != "" {
				acc.arguments += tc.Function.Arguments
				events <- stream.Event{
					Type: stream.EventToolInputDelta,
					Data: stream.ToolInputDeltaEvent{ID: acc.id, Delta: tc.Function.Arguments},
				}
			}
		}
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if finishReason == "" {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: response stream ended without a finish reason", goaierrors.ErrInvalidResponse)}}
		return
	}

	if reasoningStarted {
		events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning-0"}}
	}

	// End text if started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// Process completed tool calls
	toolCallIndices := make([]int, 0, len(currentToolCalls))
	for index := range currentToolCalls {
		toolCallIndices = append(toolCallIndices, index)
	}
	sort.Ints(toolCallIndices)
	for _, index := range toolCallIndices {
		acc := currentToolCalls[index]
		if !acc.started {
			// function.name never arrived for this tool-call index. Mirrors
			// ai-sdk's processDelta which raises AI_InvalidResponseDataError
			// in the same situation (PR #14760).
			events <- stream.Event{
				Type: stream.EventError,
				Data: stream.ErrorEvent{Error: fmt.Errorf("%w: tool call at index %d has no function.name or id", goaierrors.ErrInvalidResponse, acc.index)},
			}
			return
		}

		events <- stream.Event{
			Type: stream.EventToolInputEnd,
			Data: stream.ToolInputEndEvent{ID: acc.id},
		}

		events <- stream.Event{
			Type: stream.EventToolCall,
			Data: stream.ToolCallEvent{
				ToolCallID: acc.id,
				ToolName:   acc.name,
				Input:      json.RawMessage(acc.arguments),
			},
		}

		// Note: Tool execution is handled by goai.go's executeTools function,
		// not here in the provider. The provider just emits ToolCallEvent.
	}

	var providerMetadata map[string]any
	if usageRaw != nil {
		providerMetadata = map[string]any{
			"openaiCompat": map[string]any{
				"usageRaw": usageRaw,
			},
		}
	}

	// Emit finish step
	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}

	// Emit finish
	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		},
	}
}

type toolCallAccumulator struct {
	index     int
	id        string
	name      string
	arguments string
	started   bool // true once tool-input-start has been emitted (after function.name arrives)
}

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "stop":
		return stream.FinishReasonStop
	case "length":
		return stream.FinishReasonLength
	case "content_filter":
		return stream.FinishReasonContentFilter
	case "tool_calls":
		return stream.FinishReasonToolCalls
	default:
		return stream.FinishReasonOther
	}
}
