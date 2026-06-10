// Package azure provides an Azure OpenAI provider implementation.
package azure

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
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Options contains configuration for the Azure OpenAI provider.
type Options struct {
	APIKey         string
	ResourceName   string // Azure resource name
	DeploymentName string // Optional default deployment
	APIVersion     string // API version, defaults to 2024-02-15-preview
	BaseURL        string // Optional: override base URL (for testing)
	Headers        map[string]string

	// TokenProvider returns a Microsoft Entra ID (formerly Azure Active
	// Directory) bearer token, invoked on every request. When set, requests
	// carry "Authorization: Bearer <token>" instead of the "api-key" header.
	// Mutually exclusive with APIKey. Mirrors ai-sdk's azureADTokenProvider
	// (references/ai-sdk/packages/azure/src/azure-openai-provider.ts).
	TokenProvider func() (string, error)
}

// Provider implements the Azure OpenAI provider.
type Provider struct {
	opts Options
}

// New creates a new Azure OpenAI provider.
func New(opts Options) *Provider {
	if opts.APIKey != "" && opts.TokenProvider != nil {
		panic("azure: provide only one of APIKey or TokenProvider, not both")
	}
	if opts.APIVersion == "" {
		opts.APIVersion = "2024-02-15-preview"
	}
	return &Provider{opts: opts}
}

// setAuth applies the provider's authentication header to req: a Microsoft
// Entra ID bearer token when TokenProvider is set, otherwise the api-key
// header.
func (p *Provider) setAuth(req *http.Request) error {
	if p.opts.TokenProvider != nil {
		token, err := p.opts.TokenProvider()
		if err != nil {
			return fmt.Errorf("azure token provider: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	req.Header.Set("api-key", p.opts.APIKey)
	return nil
}

func (p *Provider) ID() string { return "azure" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	deploymentName := modelID
	if p.opts.DeploymentName != "" && modelID == "" {
		deploymentName = p.opts.DeploymentName
	}
	return &AzureLanguageModel{
		id:         modelID,
		deployment: deploymentName,
		provider:   p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &AzureImageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &AzureEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &AzureSpeechModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &AzureTranscriptionModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) Models() []string {
	return []string{
		"gpt-4o",
		"gpt-4o-mini",
		"gpt-4-turbo",
		"gpt-4",
		"gpt-35-turbo",
		"text-embedding-ada-002",
		"text-embedding-3-small",
		"text-embedding-3-large",
		"dall-e-3",
		"whisper",
		"tts",
		"tts-hd",
	}
}

func (p *Provider) baseURL(deployment string) string {
	if p.opts.BaseURL != "" {
		return p.opts.BaseURL
	}
	return fmt.Sprintf("https://%s.openai.azure.com/openai/deployments/%s",
		p.opts.ResourceName, deployment)
}

var _ provider.Provider = (*Provider)(nil)

// AzureLanguageModel implements the LanguageModel interface.
type AzureLanguageModel struct {
	id         string
	deployment string
	provider   *Provider
}

func (m *AzureLanguageModel) ID() string       { return m.id }
func (m *AzureLanguageModel) Provider() string { return "azure" }

func (m *AzureLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *AzureLanguageModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Build request body (OpenAI compatible format)
	msgs := make([]map[string]any, 0, len(options.Messages))
	for _, msg := range options.Messages {
		msgs = append(msgs, convertMessage(msg))
	}

	reqBody := map[string]any{
		"messages": msgs,
		"stream":   true,
	}

	if options.MaxOutputTokens != nil {
		reqBody["max_tokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		reqBody["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		reqBody["top_p"] = *options.TopP
	}
	if len(options.StopSequences) > 0 {
		reqBody["stop"] = options.StopSequences
	}

	// Add tools if present
	if len(options.Tools) > 0 {
		tools := make([]map[string]any, 0)
		for _, t := range options.Tools {
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  json.RawMessage(t.InputSchema),
				},
			})
		}
		reqBody["tools"] = tools
	}

	// Apply provider-specific options
	chatOpts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("invalid provider options: %w", err)}}
		return
	}
	if chatOpts.ReasoningEffort != "" {
		reqBody["reasoning_effort"] = chatOpts.ReasoningEffort
	}
	if chatOpts.ReasoningSummary != "" {
		reqBody["reasoning_summary"] = chatOpts.ReasoningSummary
	}
	if chatOpts.Store != nil {
		reqBody["store"] = *chatOpts.Store
	}
	if chatOpts.User != "" {
		reqBody["user"] = chatOpts.User
	}
	if chatOpts.ParallelToolCalls != nil {
		reqBody["parallel_tool_calls"] = *chatOpts.ParallelToolCalls
	}

	// Map ResponseFormat. Azure follows OpenAI chat semantics: json_schema if
	// a schema is provided, otherwise json_object.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		if len(options.ResponseFormat.Schema) > 0 {
			name := options.ResponseFormat.Name
			if name == "" {
				name = "response"
			}
			strict := false
			if chatOpts.StrictJsonSchema != nil {
				strict = *chatOpts.StrictJsonSchema
			}
			jsonSchema := map[string]any{
				"name":   name,
				"schema": json.RawMessage(options.ResponseFormat.Schema),
				"strict": strict,
			}
			if options.ResponseFormat.Description != "" {
				jsonSchema["description"] = options.ResponseFormat.Description
			}
			reqBody["response_format"] = map[string]any{
				"type":        "json_schema",
				"json_schema": jsonSchema,
			}
		} else {
			reqBody["response_format"] = map[string]any{"type": "json_object"}
		}
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf("%s/chat/completions?api-version=%s",
		m.provider.baseURL(m.deployment), m.provider.opts.APIVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	if err := m.provider.setAuth(req); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{
			Type: stream.EventError,
			Data: stream.ErrorEvent{Error: fmt.Errorf("Azure API error (status %d): %s", resp.StatusCode, string(body))},
		}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func convertMessage(msg message.Message) map[string]any {
	result := map[string]any{
		"role": string(msg.Role),
	}

	// Handle simple text content
	if msg.Content.Text != "" && !msg.Content.IsMultiPart() {
		result["content"] = msg.Content.Text
		return result
	}

	// Multi-part content
	parts := make([]map[string]any, 0, len(msg.Content.Parts))
	for _, part := range msg.Content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			parts = append(parts, map[string]any{
				"type": "text",
				"text": p.Text,
			})
		case message.FilePart:
			// Azure chat completions accepts images via image_url.
			if !strings.HasPrefix(p.MimeType, "image/") {
				continue
			}
			var url string
			switch d := p.Data.(type) {
			case message.FileDataBytes:
				url = "data:" + p.MimeType + ";base64," + d.Data
			case message.FileDataURL:
				url = d.URL
			default:
				continue
			}
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": url},
			})
		case message.ToolCallPart:
			// Tool calls go in tool_calls array, not content
			if result["tool_calls"] == nil {
				result["tool_calls"] = []map[string]any{}
			}
			toolCalls := result["tool_calls"].([]map[string]any)
			toolCalls = append(toolCalls, map[string]any{
				"id":   p.ID,
				"type": "function",
				"function": map[string]any{
					"name":      p.Name,
					"arguments": string(p.Input),
				},
			})
			result["tool_calls"] = toolCalls
		case message.ToolResultPart:
			result["tool_call_id"] = p.ToolCallID
			result["content"] = message.ToolOutputWire(p.Output)
			return result
		}
	}

	if len(parts) > 0 {
		result["content"] = parts
	}

	return result
}

func (m *AzureLanguageModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var currentToolCalls = make(map[int]*toolCallAccumulator)
	var usage stream.Usage
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

		var chunk streamChunk
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

		// Handle tool calls
		for _, tc := range delta.ToolCalls {
			acc, exists := currentToolCalls[tc.Index]
			if !exists {
				acc = &toolCallAccumulator{index: tc.Index}
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

	// Emit finish step
	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{
			FinishReason: finishReason,
			Usage:        usage,
		},
	}

	// Emit finish
	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{
			FinishReason: finishReason,
			Usage:        usage,
		},
	}
}

type toolCallAccumulator struct {
	index     int
	id        string
	name      string
	arguments string
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
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
