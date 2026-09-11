// Package vertex provides a Google Vertex AI provider implementation.
package vertex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/google"
	goairesponse "github.com/airlockrun/goai/response"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Options contains configuration for the Vertex AI provider.
type Options struct {
	ProjectID   string
	Location    string // e.g., "us-central1"
	AccessToken string // OAuth2 access token
	Headers     map[string]string
	// BaseURL overrides the computed Vertex endpoint. When empty, the
	// provider uses https://<location>-aiplatform.googleapis.com/v1/projects/<project>/locations/<location>.
	// Primarily useful for tests pointing at httptest.Server.
	BaseURL string
}

// Provider implements the Vertex AI provider.
type Provider struct {
	opts Options
}

// New creates a new Vertex AI provider.
func New(opts Options) *Provider {
	if opts.Location == "" {
		opts.Location = "us-central1"
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string { return "vertex" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return &VertexLanguageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &VertexImageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &VertexEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &VertexSpeechModel{id: modelID, provider: p}
}
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &VertexTranscriptionModel{id: modelID, provider: p}
}
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) baseURL() string {
	if p.opts.BaseURL != "" {
		return p.opts.BaseURL
	}
	return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s",
		vertexHost(p.opts.Location), p.opts.ProjectID, p.opts.Location)
}

// vertexHost maps a Vertex location to its API host. "global" uses the
// region-less host, the "eu" and "us" multi-region locations use the
// dedicated `.rep.` hosts, and every other location is region-prefixed.
// Mirrors ai-sdk's getHost (#b70f6ec).
func vertexHost(location string) string {
	switch location {
	case "global":
		return "aiplatform.googleapis.com"
	case "eu", "us":
		return fmt.Sprintf("aiplatform.%s.rep.googleapis.com", location)
	default:
		return fmt.Sprintf("%s-aiplatform.googleapis.com", location)
	}
}

var _ provider.Provider = (*Provider)(nil)

// VertexLanguageModel implements the LanguageModel interface.
type VertexLanguageModel struct {
	id       string
	provider *Provider
}

func (m *VertexLanguageModel) ID() string       { return m.id }
func (m *VertexLanguageModel) Provider() string { return "vertex" }

func (m *VertexLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *VertexLanguageModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	var warnings []stream.Warning

	// Build request body
	contents := make([]map[string]any, 0)
	var systemInstruction string

	for _, msg := range options.Messages {
		if msg.Role == message.RoleSystem {
			systemInstruction = msg.Content.Text
			continue
		}
		contents = append(contents, convertMessage(msg))
	}

	reqBody := map[string]any{
		"contents": contents,
	}

	if systemInstruction != "" {
		reqBody["systemInstruction"] = map[string]any{
			"parts": []map[string]any{
				{"text": systemInstruction},
			},
		}
	}

	// Generation config
	genConfig := map[string]any{}
	if options.Seed != nil {
		genConfig["seed"] = *options.Seed
	}
	if options.FrequencyPenalty != nil {
		genConfig["frequencyPenalty"] = *options.FrequencyPenalty
	}
	if options.PresencePenalty != nil {
		genConfig["presencePenalty"] = *options.PresencePenalty
	}
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		genConfig["responseMimeType"] = "application/json"
		if options.ProviderOptions["structuredOutputs"] != false && len(options.ResponseFormat.Schema) > 0 {
			genConfig["responseSchema"] = google.ResponseSchema(options.ResponseFormat.Schema)
		}
	}
	if options.MaxOutputTokens != nil {
		genConfig["maxOutputTokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		genConfig["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		genConfig["topP"] = *options.TopP
	}
	if options.TopK != nil {
		genConfig["topK"] = *options.TopK
	}
	if len(options.StopSequences) > 0 {
		genConfig["stopSequences"] = options.StopSequences
	}
	if len(genConfig) > 0 {
		reqBody["generationConfig"] = genConfig
	}

	thinking, err := google.ThinkingConfiguration(m.id, options.Reasoning, options.ProviderOptions["thinkingConfig"])
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	warnings = append(warnings, google.ThinkingConfigurationWarnings(options.Reasoning, options.ProviderOptions["thinkingConfig"], thinking)...)
	for _, key := range []string{"safetySettings", "cachedContent", "labels"} {
		if value, ok := options.ProviderOptions[key]; ok {
			reqBody[key] = value
		}
	}
	for _, key := range []string{"thinkingConfig", "responseModalities", "audioTimestamp", "mediaResolution", "imageConfig"} {
		if value, ok := options.ProviderOptions[key]; ok {
			genConfig[key] = value
		}
	}
	if thinking != nil {
		genConfig["thinkingConfig"] = thinking
	}
	if len(genConfig) > 0 {
		reqBody["generationConfig"] = genConfig
	}
	if tier, ok := options.ProviderOptions["serviceTier"].(string); ok {
		if tier != "" {
			warnings = append(warnings, stream.UnsupportedWarning("serviceTier", "Use sharedRequestType on Vertex AI."))
		}
	}
	if threshold, ok := options.ProviderOptions["threshold"].(string); ok && reqBody["safetySettings"] == nil {
		settings := []map[string]any{}
		for _, category := range []string{"HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_CIVIC_INTEGRITY"} {
			settings = append(settings, map[string]any{"category": category, "threshold": threshold})
		}
		reqBody["safetySettings"] = settings
	}
	if value, ok := options.ProviderOptions["retrievalConfig"]; ok {
		reqBody["toolConfig"] = map[string]any{"retrievalConfig": value}
	}
	if len(options.Tools) > 0 {
		reqBody["tools"] = google.PrepareTools(options.Tools, m.id)
	}
	if choice := google.ToolConfig(options.ToolChoice); choice != nil {
		if existing, ok := reqBody["toolConfig"].(map[string]any); ok {
			choice["retrievalConfig"] = existing["retrievalConfig"]
		}
		reqBody["toolConfig"] = choice
	}
	if options.ProviderOptions["streamFunctionCallArguments"] == true {
		config, ok := reqBody["toolConfig"].(map[string]any)
		if !ok {
			config = map[string]any{}
			reqBody["toolConfig"] = config
		}
		calling, ok := config["functionCallingConfig"].(map[string]any)
		if !ok {
			calling = map[string]any{}
			config["functionCallingConfig"] = calling
		}
		calling["streamFunctionCallArguments"] = true
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf("%s/publishers/google/models/%s:streamGenerateContent?alt=sse",
		m.provider.baseURL(), m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.AccessToken)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}
	for key, header := range map[string]string{"sharedRequestType": "X-Vertex-AI-LLM-Shared-Request-Type", "requestType": "X-Vertex-AI-LLM-Request-Type"} {
		if value, ok := options.ProviderOptions[key].(string); ok {
			req.Header.Set(header, value)
		}
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Vertex AI request failed", URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBytes),
			Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Vertex AI error: " + string(body), URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBytes),
			StatusCode: resp.StatusCode, ResponseHeaders: goairesponse.ExtractResponseHeaders(resp), ResponseBody: string(body),
		})}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func convertMessage(msg message.Message) map[string]any {
	role := "user"
	if msg.Role == message.RoleAssistant {
		role = "model"
	}

	parts := make([]map[string]any, 0)

	if msg.Content.Text != "" && !msg.Content.IsMultiPart() {
		parts = append(parts, map[string]any{"text": msg.Content.Text})
	} else {
		for _, part := range msg.Content.Parts {
			switch p := part.(type) {
			case message.TextPart:
				entry := map[string]any{"text": p.Text}
				if signature := google.ThoughtSignature(p.ProviderOptions); signature != "" {
					entry["thoughtSignature"] = signature
				}
				parts = append(parts, entry)
			case message.ReasoningPart:
				entry := map[string]any{"text": p.Text, "thought": true}
				if signature := google.ThoughtSignature(p.ProviderOptions); signature != "" {
					entry["thoughtSignature"] = signature
				}
				parts = append(parts, entry)
			case message.FilePart:
				switch d := p.Data.(type) {
				case message.FileDataBytes:
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": p.MimeType,
							"data":     d.Data,
						},
					})
				case message.FileDataURL:
					parts = append(parts, map[string]any{
						"fileData": map[string]any{
							"mimeType": p.MimeType,
							"fileUri":  d.URL,
						},
					})
				}
			case message.ToolCallPart:
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{
						"name": p.Name,
						"args": json.RawMessage(p.Input),
					},
				})
				if signature := google.ThoughtSignature(p.ProviderOptions); signature != "" {
					parts[len(parts)-1]["thoughtSignature"] = signature
				}
			case message.ToolResultPart:
				parts = append(parts, map[string]any{
					"functionResponse": map[string]any{
						"name":     p.ToolName,
						"response": map[string]any{"result": message.ToolOutputWire(p.Output)},
					},
				})
			}
		}
	}

	return map[string]any{
		"role":  role,
		"parts": parts,
	}
}

func (m *VertexLanguageModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var reasoningStarted bool
	var terminal bool
	var active *stream.ToolCallEvent
	var activeArgs any
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
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}

		if includeRawChunks {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: data}}
		}

		var chunk vertexStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)}}
			return
		}
		if chunk.Error != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: chunk.Error.Message, StatusCode: chunk.Error.Code, ResponseBody: data})}}
			return
		}
		if chunk.PromptFeedback.BlockReason != "" {
			terminal = true
			finishReason = stream.FinishReasonContentFilter
		}

		// Handle usage. Cached prompt tokens split onto InputTokens.CacheRead
		// and thoughts (reasoning) onto OutputTokens.Reasoning; the output
		// total is candidates + thoughts. Mirrors ai-sdk's convert-google-usage.
		if chunk.UsageMetadata.TotalTokenCount > 0 {
			um := chunk.UsageMetadata
			usage = stream.Usage{
				InputTokens:  stream.InputTokens{Total: stream.IntPtr(um.PromptTokenCount)},
				OutputTokens: stream.OutputTokens{Total: stream.IntPtr(um.CandidatesTokenCount + um.ThoughtsTokenCount), Text: stream.IntPtr(um.CandidatesTokenCount)},
			}
			if um.CachedContentTokenCount > 0 {
				usage.InputTokens.CacheRead = stream.IntPtr(um.CachedContentTokenCount)
				usage.InputTokens.NoCache = stream.IntPtr(um.PromptTokenCount - um.CachedContentTokenCount)
			} else {
				usage.InputTokens.NoCache = stream.IntPtr(um.PromptTokenCount)
			}
			if um.ThoughtsTokenCount > 0 {
				usage.OutputTokens.Reasoning = stream.IntPtr(um.ThoughtsTokenCount)
			}
		}

		if len(chunk.Candidates) == 0 {
			continue
		}

		candidate := chunk.Candidates[0]

		// Handle finish reason
		if candidate.FinishReason != "" {
			terminal = true
			switch candidate.FinishReason {
			case "STOP":
				finishReason = stream.FinishReasonStop
			case "MAX_TOKENS":
				finishReason = stream.FinishReasonLength
			case "SAFETY", "RECITATION", "IMAGE_SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
				finishReason = stream.FinishReasonContentFilter
			default:
				finishReason = stream.FinishReasonOther
			}
		}

		// Process parts
		for _, part := range candidate.Content.Parts {
			if part.Thought {
				if textStarted {
					events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
					textStarted = false
				}
				if !reasoningStarted {
					reasoningStarted = true
					events <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: "reasoning"}}
				}
				metadata := map[string]any(nil)
				if part.ThoughtSignature != "" {
					metadata = map[string]any{"google": map[string]any{"thoughtSignature": part.ThoughtSignature}}
				}
				events <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: "reasoning", Text: part.Text, ProviderMetadata: metadata}}
				continue
			}
			if part.Text != "" {
				if reasoningStarted {
					events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning"}}
					reasoningStarted = false
				}
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				metadata := map[string]any(nil)
				if part.ThoughtSignature != "" {
					metadata = map[string]any{"google": map[string]any{"thoughtSignature": part.ThoughtSignature}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: part.Text, ProviderMetadata: metadata}}
			}

			if part.FunctionCall != nil && (part.FunctionCall.PartialArgs != nil || part.FunctionCall.WillContinue || active != nil) {
				call := part.FunctionCall
				if call.Name != "" {
					if active != nil {
						events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: overlapping partial tool calls", goaierrors.ErrInvalidResponse)}}
						return
					}
					id := call.ID
					if id == "" {
						id = fmt.Sprintf("call_%s", call.Name)
					}
					active = &stream.ToolCallEvent{ToolCallID: id, ToolName: call.Name}
					activeArgs = map[string]any{}
					events <- stream.Event{Type: stream.EventToolInputStart, Data: stream.ToolInputStartEvent{ID: id, ToolName: call.Name}}
				}
				if active == nil {
					events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: partial arguments without a tool call", goaierrors.ErrInvalidResponse)}}
					return
				}
				if part.ThoughtSignature != "" {
					active.ProviderMetadata = map[string]any{"google": map[string]any{"thoughtSignature": part.ThoughtSignature}}
				}
				continuing := call.WillContinue
				for _, arg := range call.PartialArgs {
					var value any
					switch {
					case arg.StringValue != nil:
						value = *arg.StringValue
					case arg.NumberValue != nil:
						value = *arg.NumberValue
					case arg.BoolValue != nil:
						value = *arg.BoolValue
					}
					if !strings.HasPrefix(arg.JSONPath, "$.") {
						events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.ErrInvalidResponse}}
						return
					}
					var err error
					activeArgs, err = setPartialArg(activeArgs, strings.TrimPrefix(arg.JSONPath, "$."), value)
					if err != nil {
						events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)}}
						return
					}
					continuing = continuing || arg.WillContinue
				}
				if !continuing {
					input, _ := json.Marshal(activeArgs)
					active.Input = input
					events <- stream.Event{Type: stream.EventToolInputDelta, Data: stream.ToolInputDeltaEvent{ID: active.ToolCallID, Delta: string(input)}}
					events <- stream.Event{Type: stream.EventToolInputEnd, Data: stream.ToolInputEndEvent{ID: active.ToolCallID}}
					events <- stream.Event{Type: stream.EventToolCall, Data: *active}
					active = nil
					finishReason = stream.FinishReasonToolCalls
				}
				continue
			}
			if part.FunctionCall != nil && part.FunctionCall.Name != "" {
				if part.FunctionCall.Args == nil {
					part.FunctionCall.Args = map[string]any{}
				}
				argsBytes, _ := json.Marshal(part.FunctionCall.Args)
				toolCallID := fmt.Sprintf("call_%s", part.FunctionCall.Name)
				if part.FunctionCall.ID != "" {
					toolCallID = part.FunctionCall.ID
				}
				metadata := map[string]any(nil)
				if part.ThoughtSignature != "" {
					metadata = map[string]any{"google": map[string]any{"thoughtSignature": part.ThoughtSignature}}
				}

				events <- stream.Event{
					Type: stream.EventToolCall,
					Data: stream.ToolCallEvent{
						ToolCallID:       toolCallID,
						ToolName:         part.FunctionCall.Name,
						Input:            argsBytes,
						ProviderMetadata: metadata,
					},
				}

				// Note: Tool execution is handled by goai.go's executeTools function,
				// not here in the provider. The provider just emits ToolCallEvent.

				finishReason = stream.FinishReasonToolCalls
			}
		}
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	if !terminal || active != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: Vertex stream ended without a finish reason", goaierrors.ErrInvalidResponse)}}
		return
	}
	if reasoningStarted {
		events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning"}}
	}
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
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

type vertexStreamChunk struct {
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Candidates []struct {
		Content struct {
			Parts []struct {
				Thought          bool   `json:"thought"`
				ThoughtSignature string `json:"thoughtSignature"`
				Text             string `json:"text"`
				FunctionCall     *struct {
					PartialArgs  []partialArg   `json:"partialArgs"`
					WillContinue bool           `json:"willContinue"`
					ID           string         `json:"id"`
					Name         string         `json:"name"`
					Args         map[string]any `json:"args"`
				} `json:"functionCall"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	} `json:"usageMetadata"`
}

// VertexEmbeddingModel implements the EmbeddingModel interface.
type VertexEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *VertexEmbeddingModel) ID() string       { return m.id }
func (m *VertexEmbeddingModel) Provider() string { return "vertex" }
func (m *VertexEmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.id == "gemini-embedding-2" || m.id == "gemini-embedding-2-preview" {
		return 1
	}
	return 250
}
func (m *VertexEmbeddingModel) Dimensions() int { return 0 }

func (m *VertexEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	if len(opts.Values) == 0 || len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf("embedding value count must be between 1 and %d", m.MaxEmbeddingsPerCall())
	}
	instances := make([]map[string]any, len(opts.Values))
	for i, text := range opts.Values {
		instances[i] = map[string]any{"content": text}
		for _, key := range []string{"taskType", "title"} {
			if value, ok := opts.ProviderOptions[key]; ok {
				wire := key
				if key == "taskType" {
					wire = "task_type"
				}
				instances[i][wire] = value
			}
		}
	}

	reqBody := map[string]any{
		"instances": instances,
	}
	params := map[string]any{}
	if opts.Dimensions != nil {
		params["outputDimensionality"] = *opts.Dimensions
	}
	for _, key := range []string{"outputDimensionality", "autoTruncate"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			params[key] = value
		}
	}
	reqBody["parameters"] = params
	action := "predict"
	if m.MaxEmbeddingsPerCall() == 1 {
		action = "embedContent"
		for _, key := range []string{"taskType", "title"} {
			if value, ok := opts.ProviderOptions[key]; ok {
				params[key] = value
			}
		}
		reqBody = map[string]any{"content": map[string]any{"parts": []any{map[string]any{"text": opts.Values[0]}}}, "embedContentConfig": params}
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/publishers/google/models/%s:%s", m.provider.baseURL(), m.id, action)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.AccessToken)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Vertex AI error (status %d): %s", resp.StatusCode, string(body))
	}

	var embResp struct {
		UsageMetadata struct {
			PromptTokenCount int `json:"promptTokenCount"`
		} `json:"usageMetadata"`
		Embedding struct {
			Values []float64 `json:"values"`
		} `json:"embedding"`
		Predictions []struct {
			Embeddings struct {
				Values     []float64 `json:"values"`
				Statistics struct {
					TokenCount int `json:"token_count"`
				} `json:"statistics"`
			} `json:"embeddings"`
		} `json:"predictions"`
	}
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	if action == "embedContent" {
		if len(embResp.Embedding.Values) == 0 {
			return nil, goaierrors.ErrInvalidResponse
		}
		return &model.EmbedResult{Embeddings: []model.Embedding{{Values: embResp.Embedding.Values}}, Usage: model.EmbeddingUsage{Tokens: embResp.UsageMetadata.PromptTokenCount}, Response: model.EmbeddingResponse{Model: m.id}}, nil
	}

	embeddings := make([]model.Embedding, len(embResp.Predictions))
	if len(embResp.Predictions) != len(opts.Values) {
		return nil, goaierrors.ErrInvalidResponse
	}
	tokens := 0
	for i, pred := range embResp.Predictions {
		if len(pred.Embeddings.Values) == 0 {
			return nil, goaierrors.ErrInvalidResponse
		}
		tokens += pred.Embeddings.Statistics.TokenCount
		embeddings[i] = model.Embedding{
			Values: pred.Embeddings.Values,
			Index:  i,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage:      model.EmbeddingUsage{Tokens: tokens},
		Response: model.EmbeddingResponse{
			Model: m.id,
		},
	}, nil
}

// VertexImageModel implements the ImageModel interface.
type VertexImageModel struct {
	id       string
	provider *Provider
}

func (m *VertexImageModel) ID() string       { return m.id }
func (m *VertexImageModel) Provider() string { return "vertex" }
func (m *VertexImageModel) MaxImagesPerCall() int {
	if isGeminiImageModel(m.id) {
		return 10
	}
	return 4
}

func (m *VertexImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	if isGeminiImageModel(m.id) {
		return m.generateGemini(ctx, opts)
	}
	return m.generateImagen(ctx, opts)
}

func isGeminiImageModel(id string) bool {
	return strings.HasPrefix(id, "gemini-")
}

// detectVertexImageMime mirrors xai's detectImageMime helper. Duplicated
// locally because cross-package sharing of these ~15 LOC helpers is
// awkward — see goai/AGENTS.md's "mirror ai-sdk exactly" and F1 notes.
func detectVertexImageMime(data []byte) string {
	switch {
	case len(data) >= 4 && bytes.HasPrefix(data, []byte{0x89, 0x50, 0x4E, 0x47}):
		return "image/png"
	case len(data) >= 3 && bytes.HasPrefix(data, []byte{0xFF, 0xD8, 0xFF}):
		return "image/jpeg"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	case len(data) >= 4 && bytes.HasPrefix(data, []byte("GIF8")):
		return "image/gif"
	case len(data) >= 2 && bytes.HasPrefix(data, []byte("BM")):
		return "image/bmp"
	default:
		return "image/png"
	}
}

func (m *VertexImageModel) generateImagen(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	n := 1
	if opts.N > 0 {
		n = opts.N
	}

	var warnings []stream.Warning
	if opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}

	reqBody := map[string]any{
		"instances": []map[string]any{
			{"prompt": opts.Prompt},
		},
		"parameters": map[string]any{
			"sampleCount": n,
		},
	}

	// Add provider options
	params := reqBody["parameters"].(map[string]any)
	for k, v := range opts.ProviderOptions {
		params[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/publishers/google/models/%s:predict", m.provider.baseURL(), m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.AccessToken)
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Vertex AI error (status %d): %s", resp.StatusCode, string(body))
	}

	var imgResp struct {
		Predictions []struct {
			BytesBase64Encoded string `json:"bytesBase64Encoded"`
			MimeType           string `json:"mimeType"`
		} `json:"predictions"`
	}
	if err := json.Unmarshal(body, &imgResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	images := make([]model.GeneratedImage, len(imgResp.Predictions))
	for i, pred := range imgResp.Predictions {
		mimeType := pred.MimeType
		if mimeType == "" {
			mimeType = "image/png"
		}
		images[i] = model.GeneratedImage{
			Base64:   pred.BytesBase64Encoded,
			MimeType: mimeType,
		}
	}

	return &model.ImageResult{
		Images:   images,
		Warnings: warnings,
		Response: model.ImageResponse{
			Model: m.id,
		},
	}, nil
}

func (m *VertexImageModel) generateGemini(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	if opts.Mask != nil {
		return nil, errors.New("Gemini image models do not support mask-based image editing")
	}
	if opts.N > 1 {
		return nil, errors.New("Gemini image models do not support generating a set number of images per call")
	}

	var warnings []stream.Warning
	if opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}

	parts := []map[string]any{{"text": opts.Prompt}}
	for _, file := range opts.Files {
		parts = append(parts, map[string]any{
			"inlineData": map[string]any{
				"mimeType": detectVertexImageMime(file),
				"data":     base64.StdEncoding.EncodeToString(file),
			},
		})
	}

	generationConfig := map[string]any{
		"responseModalities": []string{"IMAGE"},
	}
	if opts.AspectRatio != "" {
		generationConfig["imageConfig"] = map[string]any{"aspectRatio": opts.AspectRatio}
	}
	if vertexOpts, ok := opts.ProviderOptions["vertex"].(map[string]any); ok {
		for k, v := range vertexOpts {
			generationConfig[k] = v
		}
	}

	reqBody := map[string]any{
		"contents": []map[string]any{{
			"role":  "user",
			"parts": parts,
		}},
		"generationConfig": generationConfig,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/publishers/google/models/%s:generateContent", m.provider.baseURL(), m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.AccessToken)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Vertex AI error (status %d): %s", resp.StatusCode, string(body))
	}

	var genResp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text       string `json:"text"`
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData,omitempty"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(body, &genResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	var images []model.GeneratedImage
	if len(genResp.Candidates) > 0 {
		for _, part := range genResp.Candidates[0].Content.Parts {
			if part.InlineData == nil {
				continue
			}
			if !strings.HasPrefix(part.InlineData.MimeType, "image/") {
				continue
			}
			images = append(images, model.GeneratedImage{
				Base64:   part.InlineData.Data,
				MimeType: part.InlineData.MimeType,
			})
		}
	}

	imageMetadata := make([]map[string]any, len(images))
	for i := range images {
		imageMetadata[i] = map[string]any{}
	}

	result := &model.ImageResult{
		Images:   images,
		Warnings: warnings,
		Response: model.ImageResponse{
			Model:     m.id,
			Timestamp: time.Now().Unix(),
		},
		ProviderMetadata: map[string]any{
			"vertex": map[string]any{
				"images": imageMetadata,
			},
		},
	}
	if genResp.UsageMetadata.TotalTokenCount > 0 {
		result.Usage = &model.ImageUsage{TotalTokens: genResp.UsageMetadata.TotalTokenCount}
	}
	return result, nil
}
