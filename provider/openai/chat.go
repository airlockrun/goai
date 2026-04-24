package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// ChatModel represents an OpenAI model using the Chat Completions API.
type ChatModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *ChatModel) ID() string {
	return m.id
}

// Provider returns "openai.chat".
func (m *ChatModel) Provider() string {
	return "openai.chat"
}

// Stream sends a streaming request to OpenAI using the Chat Completions API.
func (m *ChatModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *ChatModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
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
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	if m.provider.opts.Organization != "" {
		req.Header.Set("OpenAI-Organization", m.provider.opts.Organization)
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
			Error: fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func (m *ChatModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	chatMessages, err := convertToChatMessages(options.Messages)
	if err != nil {
		return nil, warnings, err
	}

	opts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on OpenAI Chat (ai-sdk parity).
	if options.TopK != nil {
		warnings = append(warnings, stream.UnsupportedWarning("topK", ""))
	}

	req := chatRequest{
		Model:    m.id,
		Stream:   true,
		Messages: chatMessages,
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
		req.Tools = convertToChatTools(options.Tools)
	}

	// Map ResponseFormat to chat response_format.
	// Mirrors ai-sdk openai-chat-language-model.ts:147-160.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		if len(options.ResponseFormat.Schema) > 0 {
			strict := false
			if opts.StrictJsonSchema != nil {
				strict = *opts.StrictJsonSchema
			}
			name := options.ResponseFormat.Name
			if name == "" {
				name = "response"
			}
			req.ResponseFormat = &chatResponseFormat{
				Type: "json_schema",
				JSONSchema: &chatJSONSchema{
					Name:        name,
					Description: options.ResponseFormat.Description,
					Schema:      options.ResponseFormat.Schema,
					Strict:      &strict,
				},
			}
		} else {
			req.ResponseFormat = &chatResponseFormat{Type: "json_object"}
		}
	}

	// logprobs: boolean true enables logprobs; a number N requests top-N
	// per token. Mirrors ai-sdk openai-chat-options.ts logprobs: boolean|number.
	if opts.Logprobs != nil {
		switch v := opts.Logprobs.(type) {
		case bool:
			if v {
				on := true
				req.Logprobs = &on
			}
		case int:
			if v > 0 {
				on := true
				n := v
				req.Logprobs = &on
				req.TopLogprobs = &n
			}
		case int64:
			if v > 0 {
				on := true
				n := int(v)
				req.Logprobs = &on
				req.TopLogprobs = &n
			}
		case float64:
			if v > 0 {
				on := true
				n := int(v)
				req.Logprobs = &on
				req.TopLogprobs = &n
			}
		}
	}

	// Stream options for usage
	req.StreamOptions = &chatStreamOptions{IncludeUsage: true}

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *ChatModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	var textStarted bool
	var currentToolCalls = make(map[int]*chatToolCallAccumulator)
	var usage stream.Usage
	var finishReason stream.FinishReason
	var logprobTokens []chatLogprobToken

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

		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Handle usage in final chunk
		if chunk.Usage != nil {
			usage = stream.UsageFrom(chunk.Usage.PromptTokens, chunk.Usage.CompletionTokens)
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Handle finish reason
		if choice.FinishReason != "" {
			finishReason = mapChatFinishReason(choice.FinishReason)
		}

		// Accumulate logprobs across chunks. Each streaming chunk typically
		// carries one token's worth of logprob data in choice.logprobs.content.
		if choice.Logprobs != nil && len(choice.Logprobs.Content) > 0 {
			logprobTokens = append(logprobTokens, choice.Logprobs.Content...)
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

		// Handle tool calls
		for _, tc := range delta.ToolCalls {
			acc, exists := currentToolCalls[tc.Index]
			if !exists {
				acc = &chatToolCallAccumulator{
					index: tc.Index,
				}
				currentToolCalls[tc.Index] = acc

				if tc.ID != "" {
					acc.id = tc.ID
				}
				if tc.Function.Name != "" {
					acc.name = tc.Function.Name
					events <- stream.Event{
						Type: stream.EventToolInputStart,
						Data: stream.ToolInputStartEvent{ID: acc.id, ToolName: acc.name},
					}
				}
			}

			if tc.ID != "" && acc.id == "" {
				acc.id = tc.ID
			}
			if tc.Function.Name != "" && acc.name == "" {
				acc.name = tc.Function.Name
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

	// Build provider metadata — surfaces logprobs under
	// providerMetadata.openai.logprobs (mirrors ai-sdk).
	var providerMetadata map[string]any
	if len(logprobTokens) > 0 {
		providerMetadata = map[string]any{
			"openai": map[string]any{
				"logprobs": mapLogprobTokens(logprobTokens),
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

// mapLogprobTokens converts the wire shape into a plain slice of maps so the
// provider metadata is consumable without exposing internal types.
func mapLogprobTokens(toks []chatLogprobToken) []map[string]any {
	out := make([]map[string]any, len(toks))
	for i, t := range toks {
		entry := map[string]any{
			"token":   t.Token,
			"logprob": t.Logprob,
		}
		if len(t.TopLogprobs) > 0 {
			top := make([]map[string]any, len(t.TopLogprobs))
			for j, a := range t.TopLogprobs {
				top[j] = map[string]any{"token": a.Token, "logprob": a.Logprob}
			}
			entry["topLogprobs"] = top
		}
		out[i] = entry
	}
	return out
}

type chatToolCallAccumulator struct {
	index     int
	id        string
	name      string
	arguments string
}

func mapChatFinishReason(reason string) stream.FinishReason {
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
