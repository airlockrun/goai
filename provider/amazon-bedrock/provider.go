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
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/anthropic"
	goairesponse "github.com/airlockrun/goai/response"
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
//
// Bedrock validates strict tools and native output against its own model
// capabilities, including inference profile and ARN model identifiers.
func bedrockAnthropicConfig(modelID string) anthropic.Config {
	supportsStrict := true
	for _, name := range []string{"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-fable-5", "claude-sonnet-5"} {
		if strings.Contains(modelID, name) {
			supportsStrict = false
		}
	}
	supportsNativeStructuredOutput := supportsStrict && !strings.Contains(modelID, "claude-sonnet-4-6") && !strings.Contains(modelID, "claude-haiku-4-5")
	return anthropic.Config{
		ProviderID:                     "amazon-bedrock",
		ToolBetaMap:                    bedrockToolBetaMap,
		ToolsStrict:                    true,
		SupportsStrictTools:            &supportsStrict,
		EmitBetasInBody:                true,
		SupportsNativeStructuredOutput: &supportsNativeStructuredOutput,
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

func (p *Provider) ID() string { return "amazon-bedrock" }

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
func (p *Provider) RerankingModel(modelID string) model.RerankingModel {
	return &BedrockRerankingModel{id: modelID, provider: p}
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
func (m *BedrockLanguageModel) Provider() string { return "amazon-bedrock" }

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
	id := bedrockModelName(m.id)
	isAnthropicFamily := strings.HasPrefix(id, "anthropic.")
	jsonToolInjected := false
	if isAnthropicFamily && options.ResponseFormat != nil && options.ResponseFormat.Type == "json" && len(options.ResponseFormat.Schema) > 0 {
		jsonToolInjected = true
	}

	if isAnthropicFamily {
		reqBody, warnings, err = m.buildAnthropicRequest(options)
	} else if strings.HasPrefix(id, "amazon.titan") {
		reqBody, warnings, err = m.buildTitanRequest(options)
	} else if strings.HasPrefix(id, "meta.llama") {
		reqBody, warnings, err = m.buildLlamaRequest(options)
	} else if strings.HasPrefix(id, "mistral.") {
		reqBody, warnings, err = m.buildMistralRequest(options)
	} else if strings.HasPrefix(id, "cohere.") {
		reqBody, warnings, err = m.buildCohereRequest(options)
	} else {
		reqBody, warnings, err = m.buildConverseRequest(options)
	}

	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if jsonToolInjected {
		var body struct {
			OutputConfig struct {
				Format json.RawMessage `json:"format"`
			} `json:"output_config"`
		}
		_ = json.Unmarshal(reqBody, &body)
		jsonToolInjected = len(body.OutputConfig.Format) == 0
	}

	endpoint := "invoke-with-response-stream"
	converse := !isAnthropicFamily && !strings.HasPrefix(id, "amazon.titan") && !strings.HasPrefix(id, "meta.llama") && !strings.HasPrefix(id, "mistral.") && !strings.HasPrefix(id, "cohere.")
	if converse {
		endpoint = "converse-stream"
	}
	url := fmt.Sprintf("%s/model/%s/%s", m.provider.baseURL(), escapeModelID(m.id), endpoint)

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
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Bedrock API request failed", URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Bedrock API error: " + string(body), URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			StatusCode: resp.StatusCode, ResponseHeaders: goairesponse.ExtractResponseHeaders(resp), ResponseBody: string(body),
		})}}
		return
	}

	// Process based on model type
	if converse {
		m.processConverseStream(ctx, resp.Body, events, options.IncludeRawChunks)
	} else if isAnthropicFamily {
		m.processAnthropicStream(ctx, &eventStreamReader{body: resp.Body, sse: true}, options.Tools, events, jsonToolInjected, options.IncludeRawChunks)
	} else {
		m.processGenericStream(ctx, &eventStreamReader{body: resp.Body}, events, options.IncludeRawChunks)
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

	cfg := bedrockAnthropicConfig(m.id)
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
			if chatOpts.ReasoningConfig.Type == "enabled" || chatOpts.ReasoningConfig.Type == "adaptive" {
				delete(body, "temperature")
				delete(body, "top_p")
				delete(body, "top_k")
			}
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

	if len(msgs) == 0 {
		return nil, nil, errors.New("Bedrock Cohere requires a user message")
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
	anthropic.ProcessStream(ctx, body, tools, events, jsonToolInjected, includeRawChunks)
}

func (m *BedrockLanguageModel) processGenericStream(ctx context.Context, body io.Reader, events chan<- stream.Event, includeRawChunks bool) {
	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var completed bool
	finishReason := stream.FinishReasonStop

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
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
			return
		}
		if outputs, ok := chunk["outputs"].([]any); ok && len(outputs) > 0 {
			if output, ok := outputs[0].(map[string]any); ok {
				chunk = output
			}
		}
		reason, _ := chunk["stop_reason"].(string)
		if v, ok := chunk["completionReason"].(string); ok {
			reason = v
		}
		if v, ok := chunk["finish_reason"].(string); ok {
			reason = v
		}
		if reason != "" || chunk["is_finished"] == true {
			completed = true
		}
		if reason == "length" || reason == "max_tokens" || reason == "LENGTH" {
			finishReason = stream.FinishReasonLength
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
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		var apiErr *goaierrors.APICallError
		if errors.As(scanner.Err(), &apiErr) {
			err = apiErr
		}
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	if !completed {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: errors.New("incomplete Bedrock model stream")}}
		return
	}

	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{FinishReason: finishReason},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{FinishReason: finishReason},
	}
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
	canonicalURI := strings.ReplaceAll(url.PathEscape(req.URL.EscapedPath()), "%2F", "/")
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
