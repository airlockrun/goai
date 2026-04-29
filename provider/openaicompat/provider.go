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
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

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

	// MessageConverter, when set, replaces the default message conversion.
	// See the MessageConverter type doc for details.
	MessageConverter MessageConverter

	// SupportsStructuredOutputs indicates the provider's endpoint honors the
	// OpenAI "json_schema" response_format with strict decoding. When false,
	// a schema request falls back to "json_object" plus prompt injection.
	// Mirrors ai-sdk's OpenAICompatibleChatConfig.supportsStructuredOutputs.
	SupportsStructuredOutputs bool
}

// Provider implements an OpenAI-compatible provider.
type Provider struct {
	opts Options
}

// New creates a new OpenAI-compatible provider.
func New(opts Options) *Provider {
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
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *CompatModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Build the request
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(m.provider.opts.AuthHeader, m.provider.opts.AuthPrefix+m.provider.opts.APIKey)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{
			Error: fmt.Errorf("%s API error (status %d): %s", m.provider.opts.ProviderID, resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func (m *CompatModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning
	if m.provider.opts.CallWarner != nil {
		warnings = append(warnings, m.provider.opts.CallWarner(options)...)
	}

	messages := options.Messages

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
			var strict *bool
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
		req.Tools = convertToTools(options.Tools)
	}

	// Translate goai's loose ToolChoice (bare strings or ai-sdk-shaped objects)
	// into Chat Completions' tool_choice. Most OpenAI-compatible APIs accept
	// the same shape. Mirrors ai-sdk parity:
	// packages/openai/src/chat/openai-chat-prepare-tools.ts.
	if options.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(options.ToolChoice)
	}

	// Stream options for usage
	req.StreamOptions = &streamOptions{IncludeUsage: true}

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
			body, err := json.Marshal(reqMap)
			return body, warnings, err
		}
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *CompatModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var currentToolCalls = make(map[int]*toolCallAccumulator)
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
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Handle usage
		if len(chunk.UsageRaw) > 0 {
			var typed chatUsage
			if err := json.Unmarshal(chunk.UsageRaw, &typed); err == nil {
				usage = stream.UsageFrom(typed.PromptTokens, typed.CompletionTokens)
			}
			var raw map[string]any
			if err := json.Unmarshal(chunk.UsageRaw, &raw); err == nil {
				usageRaw = raw
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

		// Handle text content
		if delta.Content != "" {
			if !textStarted {
				textStarted = true
				events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
			}
			events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: delta.Content}}
		}

		// Handle tool calls. Buffers id + arguments until function.name
		// arrives. Some openai-compatible providers send the first delta
		// without function.name and supply it on a later chunk; emitting
		// tool-input-start before the name would produce malformed events.
		// Mirrors ai-sdk PR #14760.
		for _, tc := range delta.ToolCalls {
			acc, exists := currentToolCalls[tc.Index]
			if !exists {
				acc = &toolCallAccumulator{index: tc.Index}
				currentToolCalls[tc.Index] = acc
			}
			if tc.ID != "" && acc.id == "" {
				acc.id = tc.ID
			}

			if !acc.started {
				acc.arguments += tc.Function.Arguments
				if tc.Function.Name != "" {
					acc.name = tc.Function.Name
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

	// End text if started
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// Process completed tool calls
	for _, acc := range currentToolCalls {
		if !acc.started {
			// function.name never arrived for this tool-call index. Mirrors
			// ai-sdk's processDelta which raises AI_InvalidResponseDataError
			// in the same situation (PR #14760).
			events <- stream.Event{
				Type: stream.EventError,
				Data: stream.ErrorEvent{Error: fmt.Errorf("openaicompat: tool call at index %d has no function.name", acc.index)},
			}
			continue
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
