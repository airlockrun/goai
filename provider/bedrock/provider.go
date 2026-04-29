// Package bedrock provides an Amazon Bedrock provider implementation.
package bedrock

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// bedrockToolBetaMap lists tool wire-types that require an
// anthropic_beta header to be set on Bedrock. Mirrors ai-sdk's
// BEDROCK_TOOL_BETA_MAP in packages/amazon-bedrock/src/anthropic/
// bedrock-anthropic-provider.ts.
var bedrockToolBetaMap = map[string]string{
	"bash_20241022":                   "computer-use-2024-10-22",
	"bash_20250124":                   "computer-use-2025-01-24",
	"text_editor_20241022":            "computer-use-2024-10-22",
	"text_editor_20250124":            "computer-use-2025-01-24",
	"text_editor_20250429":            "computer-use-2025-01-24",
	"text_editor_20250728":            "computer-use-2025-01-24",
	"computer_20241022":               "computer-use-2024-10-22",
	"computer_20250124":               "computer-use-2025-01-24",
	"tool_search_tool_regex_20251119": "tool-search-tool-2025-10-19",
	// BM25 is not currently supported on Bedrock, but including the beta
	// flag so that Bedrock returns a more useful error message if it's used.
	"tool_search_tool_bm25_20251119": "tool-search-tool-2025-10-19",
}

// bedrockAnthropicConfig returns the anthropic.Config that customizes the
// shared builder for Bedrock's InvokeModel endpoint (bedrock-2023-05-31
// wire version, strict function tools, in-body anthropic_beta list).
func bedrockAnthropicConfig() anthropic.Config {
	return anthropic.Config{
		ProviderID:      "bedrock",
		ToolBetaMap:     bedrockToolBetaMap,
		ToolsStrict:     true,
		EmitBetasInBody: true,
		TransformRequestBody: func(body map[string]any, betas []string) map[string]any {
			// Bedrock's InvokeModel uses the model id from the URL path, not
			// the body. Stripping here mirrors ai-sdk's bedrock-anthropic
			// transformRequestBody (references/ai-sdk/packages/amazon-bedrock/
			// src/anthropic/bedrock-anthropic-provider.ts).
			delete(body, "model")
			delete(body, "stream")
			body["anthropic_version"] = "bedrock-2023-05-31"
			return body
		},
	}
}

// Options contains configuration for the Bedrock provider.
type Options struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string // Optional, for temporary credentials
	Region          string
	Headers         map[string]string
}

// Provider implements the Amazon Bedrock provider.
type Provider struct {
	opts Options
}

// New creates a new Bedrock provider.
func New(opts Options) *Provider {
	if opts.Region == "" {
		opts.Region = "us-east-1"
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string { return "bedrock" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return &BedrockLanguageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &BedrockImageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &BedrockEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

// Models returns the Bedrock chat-model catalog. Mirrors
// ai-sdk/packages/amazon-bedrock/src/bedrock-chat-options.ts
// BedrockChatModelId. Embedding and image model IDs live on
// EmbeddingModels() and ImageModels() respectively.
func (p *Provider) Models() []string {
	return []string{
		// Amazon Titan text
		"amazon.titan-tg1-large",
		"amazon.titan-text-express-v1",
		"amazon.titan-text-lite-v1",
		// Anthropic Claude legacy
		"anthropic.claude-v2",
		"anthropic.claude-v2:1",
		"anthropic.claude-instant-v1",
		// Anthropic Claude 4.x
		"anthropic.claude-opus-4-7",
		"anthropic.claude-opus-4-6-v1",
		"anthropic.claude-sonnet-4-6-v1",
		"anthropic.claude-opus-4-5-20251101-v1:0",
		"anthropic.claude-haiku-4-5-20251001-v1:0",
		"anthropic.claude-sonnet-4-5-20250929-v1:0",
		"anthropic.claude-sonnet-4-20250514-v1:0",
		"anthropic.claude-opus-4-20250514-v1:0",
		"anthropic.claude-opus-4-1-20250805-v1:0",
		// Anthropic Claude 3.x
		"anthropic.claude-3-7-sonnet-20250219-v1:0",
		"anthropic.claude-3-5-sonnet-20240620-v1:0",
		"anthropic.claude-3-5-sonnet-20241022-v2:0",
		"anthropic.claude-3-5-haiku-20241022-v1:0",
		"anthropic.claude-3-sonnet-20240229-v1:0",
		"anthropic.claude-3-haiku-20240307-v1:0",
		"anthropic.claude-3-opus-20240229-v1:0",
		// Cohere
		"cohere.command-text-v14",
		"cohere.command-light-text-v14",
		"cohere.command-r-v1:0",
		"cohere.command-r-plus-v1:0",
		// Meta Llama
		"meta.llama3-70b-instruct-v1:0",
		"meta.llama3-8b-instruct-v1:0",
		"meta.llama3-1-405b-instruct-v1:0",
		"meta.llama3-1-70b-instruct-v1:0",
		"meta.llama3-1-8b-instruct-v1:0",
		"meta.llama3-2-11b-instruct-v1:0",
		"meta.llama3-2-1b-instruct-v1:0",
		"meta.llama3-2-3b-instruct-v1:0",
		"meta.llama3-2-90b-instruct-v1:0",
		// Mistral
		"mistral.mistral-7b-instruct-v0:2",
		"mistral.mixtral-8x7b-instruct-v0:1",
		"mistral.mistral-large-2402-v1:0",
		"mistral.mistral-small-2402-v1:0",
		// OpenAI OSS
		"openai.gpt-oss-120b-1:0",
		"openai.gpt-oss-20b-1:0",
		// US region-inference variants — Amazon Nova
		"us.amazon.nova-premier-v1:0",
		"us.amazon.nova-pro-v1:0",
		"us.amazon.nova-micro-v1:0",
		"us.amazon.nova-lite-v1:0",
		// US region-inference variants — Anthropic
		"us.anthropic.claude-3-sonnet-20240229-v1:0",
		"us.anthropic.claude-3-opus-20240229-v1:0",
		"us.anthropic.claude-3-haiku-20240307-v1:0",
		"us.anthropic.claude-3-5-sonnet-20240620-v1:0",
		"us.anthropic.claude-3-5-haiku-20241022-v1:0",
		"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		"us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		"us.anthropic.claude-opus-4-7",
		"us.anthropic.claude-opus-4-6-v1",
		"us.anthropic.claude-sonnet-4-6-v1",
		"us.anthropic.claude-opus-4-5-20251101-v1:0",
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		"us.anthropic.claude-sonnet-4-20250514-v1:0",
		"us.anthropic.claude-opus-4-20250514-v1:0",
		"us.anthropic.claude-opus-4-1-20250805-v1:0",
		"us.anthropic.claude-haiku-4-5-20251001-v1:0",
		// US region-inference variants — Meta
		"us.meta.llama3-2-11b-instruct-v1:0",
		"us.meta.llama3-2-3b-instruct-v1:0",
		"us.meta.llama3-2-90b-instruct-v1:0",
		"us.meta.llama3-2-1b-instruct-v1:0",
		"us.meta.llama3-1-8b-instruct-v1:0",
		"us.meta.llama3-1-70b-instruct-v1:0",
		"us.meta.llama3-3-70b-instruct-v1:0",
		"us.meta.llama4-scout-17b-instruct-v1:0",
		"us.meta.llama4-maverick-17b-instruct-v1:0",
		// US region-inference variants — other
		"us.deepseek.r1-v1:0",
		"us.mistral.pixtral-large-2502-v1:0",
	}
}

// EmbeddingModels returns the Bedrock embedding catalog. Mirrors
// ai-sdk/packages/amazon-bedrock/src/bedrock-embedding-options.ts
// BedrockEmbeddingModelId.
func (p *Provider) EmbeddingModels() []string {
	return []string{
		"amazon.titan-embed-text-v1",
		"amazon.titan-embed-text-v2:0",
		"cohere.embed-english-v3",
		"cohere.embed-multilingual-v3",
	}
}

// ImageModels returns the Bedrock image-generation catalog. Mirrors
// ai-sdk/packages/amazon-bedrock/src/bedrock-image-settings.ts
// BedrockImageModelId. amazon.titan-image-generator-v1 is retained for
// callers still using the older Titan image endpoint.
func (p *Provider) ImageModels() []string {
	return []string{
		"amazon.nova-canvas-v1:0",
		"amazon.titan-image-generator-v1",
	}
}

func (p *Provider) baseURL() string {
	return fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", p.opts.Region)
}

var _ provider.Provider = (*Provider)(nil)

// BedrockLanguageModel implements the LanguageModel interface.
type BedrockLanguageModel struct {
	id       string
	provider *Provider
}

func (m *BedrockLanguageModel) ID() string       { return m.id }
func (m *BedrockLanguageModel) Provider() string { return "bedrock" }

func (m *BedrockLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *BedrockLanguageModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Build request based on model type
	var reqBody []byte
	var warnings []stream.Warning
	var err error

	// The anthropic family uses synthetic-tool injection (matches Phase B in
	// goai/provider/anthropic); all other families lack a structured-output API
	// so they get prompt injection via buildXxxRequest.
	isAnthropicFamily := strings.HasPrefix(m.id, "anthropic.") || (!strings.HasPrefix(m.id, "amazon.titan") && !strings.HasPrefix(m.id, "meta.llama") && !strings.HasPrefix(m.id, "mistral.") && !strings.HasPrefix(m.id, "cohere."))
	jsonToolInjected := false
	if isAnthropicFamily && options.ResponseFormat != nil && options.ResponseFormat.Type == "json" && len(options.ResponseFormat.Schema) > 0 {
		jsonToolInjected = true
	}

	if strings.HasPrefix(m.id, "anthropic.") {
		reqBody, warnings, err = m.buildAnthropicRequest(options)
	} else if strings.HasPrefix(m.id, "amazon.titan") {
		reqBody, warnings, err = m.buildTitanRequest(options)
	} else if strings.HasPrefix(m.id, "meta.llama") {
		reqBody, warnings, err = m.buildLlamaRequest(options)
	} else if strings.HasPrefix(m.id, "mistral.") {
		reqBody, warnings, err = m.buildMistralRequest(options)
	} else if strings.HasPrefix(m.id, "cohere.") {
		reqBody, warnings, err = m.buildCohereRequest(options)
	} else {
		// Default to Anthropic-style for unknown models
		reqBody, warnings, err = m.buildAnthropicRequest(options)
	}

	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf("%s/model/%s/invoke-with-response-stream", m.provider.baseURL(), m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/vnd.amazon.eventstream")
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	// Sign request with AWS Signature Version 4
	m.signRequest(req, reqBody)

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

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
			Data: stream.ErrorEvent{Error: fmt.Errorf("Bedrock API error (status %d): %s", resp.StatusCode, string(body))},
		}
		return
	}

	// Process based on model type
	if strings.HasPrefix(m.id, "anthropic.") {
		m.processAnthropicStream(ctx, resp.Body, options.Tools, events, jsonToolInjected, options.IncludeRawChunks)
	} else {
		m.processGenericStream(ctx, resp.Body, events, options.IncludeRawChunks)
	}
}

func (m *BedrockLanguageModel) buildAnthropicRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	// Bedrock reuses the Anthropic builder with hook overrides for its
	// InvokeModel wire format. Bedrock-specific ChatOptions
	// (ReasoningConfig, ServiceTier, CacheControl, AdditionalModelRequestFields)
	// are spliced in via a per-call TransformRequestBody closure so the
	// shared builder stays Anthropic-only.
	chatOpts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid provider options: %w", err)
	}

	cfg := bedrockAnthropicConfig()
	baseTransform := cfg.TransformRequestBody
	cfg.TransformRequestBody = func(body map[string]any, betas []string) map[string]any {
		body = baseTransform(body, betas)
		if chatOpts.ReasoningConfig != nil {
			reasoning := map[string]any{
				"type": chatOpts.ReasoningConfig.Type,
			}
			if chatOpts.ReasoningConfig.BudgetTokens > 0 {
				reasoning["budget_tokens"] = chatOpts.ReasoningConfig.BudgetTokens
			}
			if chatOpts.ReasoningConfig.MaxReasoningEffort != "" {
				reasoning["max_reasoning_effort"] = chatOpts.ReasoningConfig.MaxReasoningEffort
			}
			body["thinking"] = reasoning
		}
		if chatOpts.ServiceTier != "" {
			body["service_tier"] = chatOpts.ServiceTier
		}
		if chatOpts.CacheControl != nil {
			cc := map[string]any{"type": chatOpts.CacheControl.Type}
			if chatOpts.CacheControl.TTL != "" {
				cc["ttl"] = chatOpts.CacheControl.TTL
			}
			body["cache_control"] = cc
		}
		for k, v := range chatOpts.AdditionalModelRequestFields {
			body[k] = v
		}
		return body
	}

	body, _, warnings, err := anthropic.BuildRequestBody(cfg, m.id, options)
	return body, warnings, err
}

func (m *BedrockLanguageModel) buildTitanRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	// Build prompt from messages
	var prompt strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n\n")
		case message.RoleUser:
			prompt.WriteString("User: ")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n")
		case message.RoleAssistant:
			prompt.WriteString("Bot: ")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n")
		}
	}
	prompt.WriteString("Bot:")

	reqBody := map[string]any{
		"inputText": prompt.String(),
	}

	textGenConfig := map[string]any{}
	if options.MaxOutputTokens != nil {
		textGenConfig["maxTokenCount"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		textGenConfig["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		textGenConfig["topP"] = *options.TopP
	}
	if len(options.StopSequences) > 0 {
		textGenConfig["stopSequences"] = options.StopSequences
	}

	if len(textGenConfig) > 0 {
		reqBody["textGenerationConfig"] = textGenConfig
	}

	body, err := json.Marshal(reqBody)
	return body, nil, err
}

func (m *BedrockLanguageModel) buildLlamaRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	var prompt strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			prompt.WriteString("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("<|eot_id|>")
		case message.RoleUser:
			prompt.WriteString("<|start_header_id|>user<|end_header_id|>\n\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("<|eot_id|>")
		case message.RoleAssistant:
			prompt.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("<|eot_id|>")
		}
	}
	prompt.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")

	reqBody := map[string]any{
		"prompt": prompt.String(),
	}

	if options.MaxOutputTokens != nil {
		reqBody["max_gen_len"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		reqBody["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		reqBody["top_p"] = *options.TopP
	}

	body, err := json.Marshal(reqBody)
	return body, nil, err
}

func (m *BedrockLanguageModel) buildMistralRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	var prompt strings.Builder
	prompt.WriteString("<s>")
	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			// Mistral's instruct format has no distinct system role; wrap the
			// injected JSON instruction as an [INST] block before user turns.
			prompt.WriteString("[INST] ")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString(" [/INST]")
		case message.RoleUser:
			prompt.WriteString("[INST] ")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString(" [/INST]")
		case message.RoleAssistant:
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("</s><s>")
		}
	}

	reqBody := map[string]any{
		"prompt": prompt.String(),
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

	body, err := json.Marshal(reqBody)
	return body, nil, err
}

func (m *BedrockLanguageModel) buildCohereRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	msgs := make([]map[string]any, 0, len(messages))
	var preamble string

	for _, msg := range messages {
		if msg.Role == message.RoleSystem {
			preamble = msg.Content.Text
			continue
		}
		role := "USER"
		if msg.Role == message.RoleAssistant {
			role = "CHATBOT"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"message": msg.Content.Text,
		})
	}

	reqBody := map[string]any{
		"chat_history": msgs[:len(msgs)-1],
		"message":      msgs[len(msgs)-1]["message"],
	}

	if preamble != "" {
		reqBody["preamble"] = preamble
	}
	if options.MaxOutputTokens != nil {
		reqBody["max_tokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		reqBody["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		reqBody["p"] = *options.TopP
	}

	body, err := json.Marshal(reqBody)
	return body, nil, err
}

func (m *BedrockLanguageModel) processAnthropicStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, jsonToolInjected bool, includeRawChunks bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var currentToolID, currentToolName string
	var currentToolArgs strings.Builder
	// Mirrors the anthropic provider: when the synthetic JSON tool fires,
	// stream its input as text and rewrite the stop reason to Stop.
	inSyntheticJSON := false
	syntheticJSONSeen := false
	// Track input/output counts separately so we can emit a v3 Usage
	// at the end with nil for unreported counts.
	var inputTotal, outputTotal int
	var inputReported, outputReported bool
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

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock.Type == "text" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
			} else if event.ContentBlock.Type == "tool_use" {
				// "json" mirrors the synthetic tool name in
				// goai/provider/anthropic (see syntheticJSONToolName).
				if jsonToolInjected && event.ContentBlock.Name == "json" {
					inSyntheticJSON = true
					syntheticJSONSeen = true
					if !textStarted {
						textStarted = true
						events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
					}
					break
				}
				currentToolID = event.ContentBlock.ID
				currentToolName = event.ContentBlock.Name
				currentToolArgs.Reset()
				events <- stream.Event{
					Type: stream.EventToolInputStart,
					Data: stream.ToolInputStartEvent{ID: currentToolID, ToolName: currentToolName},
				}
			}

		case "content_block_delta":
			if event.Delta.Type == "text_delta" {
				events <- stream.Event{
					Type: stream.EventTextDelta,
					Data: stream.TextDeltaEvent{Text: event.Delta.Text},
				}
			} else if event.Delta.Type == "input_json_delta" {
				if inSyntheticJSON {
					events <- stream.Event{
						Type: stream.EventTextDelta,
						Data: stream.TextDeltaEvent{Text: event.Delta.PartialJSON},
					}
					break
				}
				currentToolArgs.WriteString(event.Delta.PartialJSON)
				events <- stream.Event{
					Type: stream.EventToolInputDelta,
					Data: stream.ToolInputDeltaEvent{ID: currentToolID, Delta: event.Delta.PartialJSON},
				}
			}

		case "content_block_stop":
			if inSyntheticJSON {
				inSyntheticJSON = false
				break
			}
			if currentToolID != "" {
				events <- stream.Event{
					Type: stream.EventToolInputEnd,
					Data: stream.ToolInputEndEvent{ID: currentToolID},
				}

				events <- stream.Event{
					Type: stream.EventToolCall,
					Data: stream.ToolCallEvent{
						ToolCallID: currentToolID,
						ToolName:   currentToolName,
						Input:      json.RawMessage(currentToolArgs.String()),
					},
				}

				// Note: Tool execution is handled by goai.go's executeTools function,
				// not here in the provider. The provider just emits ToolCallEvent.

				currentToolID = ""
				currentToolName = ""
			}

		case "message_delta":
			if event.Delta.StopReason != "" {
				switch event.Delta.StopReason {
				case "end_turn":
					finishReason = stream.FinishReasonStop
				case "max_tokens":
					finishReason = stream.FinishReasonLength
				case "tool_use":
					finishReason = stream.FinishReasonToolCalls
				default:
					finishReason = stream.FinishReasonOther
				}
			}
			if event.Usage.OutputTokens > 0 {
				outputTotal = event.Usage.OutputTokens
				outputReported = true
			}

		case "message_start":
			if event.Message.Usage.InputTokens > 0 {
				inputTotal = event.Message.Usage.InputTokens
				inputReported = true
			}
		}
	}

	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	if syntheticJSONSeen && finishReason == stream.FinishReasonToolCalls {
		finishReason = stream.FinishReasonStop
	}

	var usage stream.Usage
	if inputReported {
		usage.InputTokens.Total = stream.IntPtr(inputTotal)
	}
	if outputReported {
		usage.OutputTokens.Total = stream.IntPtr(outputTotal)
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{FinishReason: finishReason, Usage: usage},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{FinishReason: finishReason, Usage: usage},
	}
}

func (m *BedrockLanguageModel) processGenericStream(ctx context.Context, body io.Reader, events chan<- stream.Event, includeRawChunks bool) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if line == "" {
			continue
		}

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: line}}
		}

		// Parse based on model-specific format
		var chunk map[string]any
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}

		// Extract text from various model formats
		var text string
		if output, ok := chunk["outputText"].(string); ok {
			text = output // Titan
		} else if generation, ok := chunk["generation"].(string); ok {
			text = generation // Llama/Mistral
		} else if t, ok := chunk["text"].(string); ok {
			text = t // Cohere
		}

		if text != "" {
			if !textStarted {
				textStarted = true
				events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
			}
			events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}}
		}
	}

	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{FinishReason: stream.FinishReasonStop},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop},
	}
}

type anthropicStreamEvent struct {
	Type         string `json:"type"`
	ContentBlock struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Message struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// AWS Signature Version 4 signing
func (m *BedrockLanguageModel) signRequest(req *http.Request, payload []byte) {
	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("X-Amz-Date", amzDate)
	if m.provider.opts.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", m.provider.opts.SessionToken)
	}

	// Create canonical request
	canonicalURI := req.URL.Path
	canonicalQueryString := req.URL.RawQuery

	signedHeaders := "content-type;host;x-amz-date"
	if m.provider.opts.SessionToken != "" {
		signedHeaders = "content-type;host;x-amz-date;x-amz-security-token"
	}

	payloadHash := sha256Hash(payload)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	canonicalHeaders := fmt.Sprintf("content-type:%s\nhost:%s\nx-amz-date:%s\n",
		req.Header.Get("Content-Type"), req.Host, amzDate)
	if m.provider.opts.SessionToken != "" {
		canonicalHeaders = fmt.Sprintf("content-type:%s\nhost:%s\nx-amz-date:%s\nx-amz-security-token:%s\n",
			req.Header.Get("Content-Type"), req.Host, amzDate, m.provider.opts.SessionToken)
	}

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	// Create string to sign
	algorithm := "AWS4-HMAC-SHA256"
	credentialScope := fmt.Sprintf("%s/%s/bedrock/aws4_request", dateStamp, m.provider.opts.Region)
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		credentialScope,
		sha256Hash([]byte(canonicalRequest)),
	}, "\n")

	// Calculate signature
	kDate := hmacSHA256([]byte("AWS4"+m.provider.opts.SecretAccessKey), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(m.provider.opts.Region))
	kService := hmacSHA256(kRegion, []byte("bedrock"))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	signature := hex.EncodeToString(hmacSHA256(kSigning, []byte(stringToSign)))

	// Add authorization header
	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, m.provider.opts.AccessKeyID, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)
}

func sha256Hash(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}
