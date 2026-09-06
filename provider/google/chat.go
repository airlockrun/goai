package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// GoogleModel represents a Google Gemini model.
type GoogleModel struct {
	id       string
	provider *Provider
}

// ID returns the model ID.
func (m *GoogleModel) ID() string {
	return m.id
}

// Provider returns "google".
func (m *GoogleModel) Provider() string {
	return "google"
}

// Stream sends a streaming request to Google AI.
func (m *GoogleModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *GoogleModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Build the request
	reqBody, warnings, err := m.buildRequest(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf("%s/%s:streamGenerateContent?key=%s&alt=sse",
		strings.TrimRight(m.provider.opts.BaseURL, "/"), modelPath(m.id), m.provider.opts.APIKey)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{Warnings: warnings}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "Google AI API request failed", URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: fmt.Sprintf("Google AI API error: %s", body), URL: req.URL.String(), RequestBodyValues: json.RawMessage(reqBody),
			StatusCode: resp.StatusCode, ResponseHeaders: responseHeaders(resp.Header), ResponseBody: string(body),
		})}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events, options.IncludeRawChunks)
}

func responseHeaders(headers http.Header) map[string]string {
	flattened := make(map[string]string, len(headers))
	for name := range headers {
		flattened[name] = headers.Get(name)
	}
	return flattened
}

func (m *GoogleModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	// Parse typed provider options
	opts, err := provider.ParseProviderOptions[GenerativeAIOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on Google (ai-sdk parity).
	isGemini25 := strings.HasPrefix(strings.ToLower(path.Base(m.id)), "gemini-2.5-")
	if options.FrequencyPenalty != nil && isGemini25 {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if options.PresencePenalty != nil && isGemini25 {
		warnings = append(warnings, stream.UnsupportedWarning("presencePenalty", ""))
	}

	var contents []geminiContent
	var systemInstruction *geminiContent

	for _, msg := range options.Messages {
		switch msg.Role {
		case message.RoleSystem:
			systemInstruction = &geminiContent{
				Parts: []geminiPart{{Text: getTextFromContent(msg.Content)}},
			}
		case message.RoleUser:
			parts, partWarnings := convertToGeminiParts(msg.Content)
			warnings = append(warnings, partWarnings...)
			contents = append(contents, geminiContent{
				Role:  "user",
				Parts: parts,
			})
		case message.RoleAssistant:
			parts := convertAssistantParts(msg.Content)
			contents = append(contents, geminiContent{
				Role:  "model",
				Parts: parts,
			})
		case message.RoleTool:
			// Tool results — emit one functionResponse part per ToolResultPart,
			// then append any attachment FilePart as sibling inline/fileData
			// parts in the same user turn. This matches ai-sdk's multimodal
			// functionResponse behavior (#47114a3): a remote file URL becomes a
			// fileData part, inline base64 becomes inlineData. An image FilePart
			// also gets a synthetic explanatory text part.
			var parts []geminiPart
			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.ToolResultPart:
					parts = append(parts, geminiPart{
						FunctionResponse: &geminiFunctionResponse{
							ID:       p.ToolCallID,
							Name:     p.ToolName,
							Response: map[string]any{"result": message.ToolOutputWire(p.Output)},
						},
					})
				case message.TextPart:
					parts = append(parts, geminiPart{
						FunctionResponse: &geminiFunctionResponse{
							ID:       toolCallIDFromMessage(msg),
							Name:     toolNameFromMessage(msg),
							Response: map[string]any{"result": p.Text},
						},
					})
				case message.FilePart:
					switch d := p.Data.(type) {
					case message.FileDataURL:
						parts = append(parts, geminiPart{
							FileData: &geminiFileData{MimeType: p.MimeType, FileURI: d.URL},
						})
						if isImageMimeType(p.MimeType) {
							parts = append(parts, geminiPart{
								Text: "Tool executed successfully and returned this image as a response",
							})
						}
					case message.FileDataBytes:
						parts = append(parts, geminiPart{
							InlineData: &geminiInlineData{MimeType: p.MimeType, Data: d.Data},
						})
						if isImageMimeType(p.MimeType) {
							parts = append(parts, geminiPart{
								Text: "Tool executed successfully and returned this image as a response",
							})
						}
					case message.FileDataText, message.FileDataReference:
						warnings = append(warnings, stream.UnsupportedWarning("filePart", "file data type not supported by Gemini"))
					}
				}
			}
			if len(parts) > 0 {
				contents = append(contents, geminiContent{
					Role:  "user",
					Parts: parts,
				})
			}
		}
	}

	req := geminiRequest{
		Contents: contents,
	}

	if systemInstruction != nil {
		req.SystemInstruction = systemInstruction
	}

	// Generation config
	config := &geminiGenerationConfig{}
	hasConfig := false
	config.Seed = options.Seed
	if options.Seed != nil {
		hasConfig = true
	}
	if !isGemini25 {
		config.FrequencyPenalty = options.FrequencyPenalty
		config.PresencePenalty = options.PresencePenalty
		if options.FrequencyPenalty != nil || options.PresencePenalty != nil {
			hasConfig = true
		}
	}

	if options.Temperature != nil {
		config.Temperature = options.Temperature
		hasConfig = true
	}
	if options.TopP != nil {
		config.TopP = options.TopP
		hasConfig = true
	}
	if options.TopK != nil {
		config.TopK = options.TopK
		hasConfig = true
	}
	if options.MaxOutputTokens != nil {
		config.MaxOutputTokens = options.MaxOutputTokens
		hasConfig = true
	}
	if len(options.StopSequences) > 0 {
		config.StopSequences = options.StopSequences
		hasConfig = true
	}

	if hasConfig {
		req.GenerationConfig = config
	}

	// Add tools (already ordered by core). Provider-defined tools (e.g.
	// googleSearch, googleMaps) emit separate entries from function tools.
	// Mirrors ai-sdk's prepare-tools behavior.
	if len(options.Tools) > 0 {
		req.Tools = prepareGeminiTools(options.Tools, m.id)
	}

	// Translate goai's loose ToolChoice (bare strings or ai-sdk-shaped objects)
	// into Gemini's toolConfig.functionCallingConfig. Mirrors ai-sdk parity:
	// packages/google/src/google-prepare-tools.ts.
	if options.ToolChoice != nil {
		req.ToolConfig = convertToolChoice(options.ToolChoice)
		// includeServerSideToolInvocations: true mirrors ai-sdk's prepare-tools
		// behavior for non-Vertex Gemini (PR #14767). The Vertex endpoint
		// rejects this field, but goai's vertex package is separate, so the
		// google package is always non-Vertex.
		if req.ToolConfig != nil {
			req.ToolConfig.IncludeServerSideToolInvocations = true
		}
	}

	// Apply provider-specific options from typed struct

	// safetySettings
	if len(opts.SafetySettings) > 0 {
		req.SafetySettings = make([]geminiSafetySetting, len(opts.SafetySettings))
		for i, s := range opts.SafetySettings {
			req.SafetySettings[i] = geminiSafetySetting{
				Category:  s.Category,
				Threshold: s.Threshold,
			}
		}
	}
	if opts.SafetySettings == nil && opts.Threshold != "" {
		for _, category := range []string{"HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_CIVIC_INTEGRITY"} {
			req.SafetySettings = append(req.SafetySettings, geminiSafetySetting{Category: category, Threshold: opts.Threshold})
		}
	}

	// cachedContent
	if opts.CachedContent != "" {
		req.CachedContent = opts.CachedContent
	}

	// thinkingConfig
	thinking, err := ThinkingConfiguration(m.id, options.Reasoning, options.ProviderOptions["thinkingConfig"])
	if err != nil {
		return nil, warnings, err
	}
	if thinking != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ThinkingConfig = thinking
	}

	// responseModalities
	if len(opts.ResponseModalities) > 0 {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ResponseModalities = opts.ResponseModalities
	}

	// audioTimestamp
	if opts.AudioTimestamp != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.AudioTimestamp = opts.AudioTimestamp
	}

	// mediaResolution
	if opts.MediaResolution != "" {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.MediaResolution = opts.MediaResolution
	}

	// ResponseFormat: JSON mode, optionally with a translated OpenAPI schema.
	// Mirrors ai-sdk packages/google/src/google-generative-ai-language-model.ts.
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ResponseMimeType = "application/json"
		// Honor googleOptions.StructuredOutputs (default: true).
		structured := true
		if opts.StructuredOutputs != nil {
			structured = *opts.StructuredOutputs
		}
		if structured && len(options.ResponseFormat.Schema) > 0 {
			if converted := convertJSONSchemaToOpenAPI(options.ResponseFormat.Schema); converted != nil {
				req.GenerationConfig.ResponseSchema = converted
			}
		}
	}

	req.ServiceTier = opts.ServiceTier
	req.Labels = opts.Labels
	if opts.ImageConfig != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ImageConfig = opts.ImageConfig
	}
	if opts.RetrievalConfig != nil {
		if req.ToolConfig == nil {
			req.ToolConfig = &geminiToolConfig{}
		}
		req.ToolConfig.RetrievalConfig = opts.RetrievalConfig
	}

	if opts.StreamFunctionCallArguments != nil && *opts.StreamFunctionCallArguments {
		warnings = append(warnings, stream.UnsupportedWarning("streamFunctionCallArguments", "Only supported by Vertex AI."))
	}
	if opts.SharedRequestType != "" {
		warnings = append(warnings, stream.UnsupportedWarning("sharedRequestType", "Only supported by Vertex AI."))
	}
	if opts.RequestType != "" {
		warnings = append(warnings, stream.UnsupportedWarning("requestType", "Only supported by Vertex AI."))
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *GoogleModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event, includeRawChunks bool) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	streamReader := goaiinternal.NewStreamReader(body)
	scanner := bufio.NewScanner(streamReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var reasoningStarted bool
	var usage stream.Usage
	var finishReason stream.FinishReason
	var pendingToolCalls []stream.ToolCallEvent
	var groundingMetadata *geminiGroundingMetadata
	var urlContextMetadata *geminiURLContextMetadata
	var serviceTier string

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

		var chunk geminiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)}}
			return
		}
		if chunk.Error != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: chunk.Error.Message, StatusCode: chunk.Error.Code, ResponseBody: data})}}
			return
		}
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			finishReason = stream.FinishReasonContentFilter
		}

		// Process candidates
		for _, candidate := range chunk.Candidates {
			// Handle finish reason
			if candidate.FinishReason != "" {
				finishReason = mapGeminiFinishReason(candidate.FinishReason)
			}

			// Accumulate grounding / url-context metadata. Gemini typically
			// emits it on the final chunk; we take the last non-nil value so
			// multi-chunk streams still land with complete data.
			if candidate.GroundingMetadata != nil {
				groundingMetadata = candidate.GroundingMetadata
			}
			if candidate.URLContextMetadata != nil {
				urlContextMetadata = candidate.URLContextMetadata
			}

			// Process content parts
			if candidate.Content != nil {
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
					// Text content
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

					// Function call. Vertex emits no-args calls as
					// `{name: "X"}` with no args; default to "{}" so the
					// downstream tool executor receives valid JSON
					// (ai-sdk #14968). thoughtSignature on the same part
					// rides through providerMetadata.google so Vertex's
					// multi-turn signature rule is satisfied on the next
					// agent step.
					if part.FunctionCall != nil {
						var inputBytes []byte
						if part.FunctionCall.Args != nil {
							inputBytes, _ = json.Marshal(part.FunctionCall.Args)
						} else {
							inputBytes = []byte("{}")
						}
						// Prefer the call id returned by the Gemini API; fall
						// back to the function name when absent. ai-sdk #15317.
						toolCallID := part.FunctionCall.ID
						if toolCallID == "" {
							toolCallID = part.FunctionCall.Name
						}
						tc := stream.ToolCallEvent{
							ToolCallID: toolCallID,
							ToolName:   part.FunctionCall.Name,
							Input:      inputBytes,
						}
						if part.ThoughtSignature != "" {
							tc.ProviderMetadata = map[string]any{
								"google": map[string]any{"thoughtSignature": part.ThoughtSignature},
							}
						}
						pendingToolCalls = append(pendingToolCalls, tc)
					}
				}
			}
		}

		// Process usage
		if chunk.UsageMetadata != nil {
			usage = chunk.UsageMetadata.toUsage()
			if chunk.UsageMetadata.ServiceTier != "" {
				serviceTier = chunk.UsageMetadata.ServiceTier
			}
		}
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	// End text if started
	if finishReason == "" {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: fmt.Errorf("%w: Gemini stream ended without a finish reason", goaierrors.ErrInvalidResponse)}}
		return
	}
	if reasoningStarted {
		events <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: "reasoning"}}
	}
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	// Process tool calls
	for _, tc := range pendingToolCalls {
		events <- stream.Event{
			Type: stream.EventToolInputStart,
			Data: stream.ToolInputStartEvent{ID: tc.ToolCallID, ToolName: tc.ToolName},
		}
		events <- stream.Event{
			Type: stream.EventToolInputEnd,
			Data: stream.ToolInputEndEvent{ID: tc.ToolCallID},
		}
		events <- stream.Event{Type: stream.EventToolCall, Data: tc}

		// Note: Tool execution is handled by goai.go's executeTools function,
		// not here in the provider. The provider just emits ToolCallEvent.
	}

	// Set finish reason if there were tool calls
	if len(pendingToolCalls) > 0 && finishReason == stream.FinishReasonStop {
		finishReason = stream.FinishReasonToolCalls
	}

	// Emit grounding sources before the finish event so they land in the
	// step's content. Mirrors ai-sdk's extractSources at
	// packages/google/src/google-language-model.ts.
	for _, src := range extractSources(groundingMetadata) {
		events <- stream.Event{Type: stream.EventSource, Data: src}
	}

	// Build provider metadata — surfaces groundingMetadata and
	// urlContextMetadata under providerMetadata.google (mirrors ai-sdk).
	var providerMetadata map[string]any
	if groundingMetadata != nil || urlContextMetadata != nil || serviceTier != "" {
		google := map[string]any{}
		if groundingMetadata != nil {
			google["groundingMetadata"] = mapGroundingMetadata(groundingMetadata)
		}
		if urlContextMetadata != nil {
			google["urlContextMetadata"] = mapURLContextMetadata(urlContextMetadata)
		}
		if serviceTier != "" {
			google["serviceTier"] = serviceTier
		}
		providerMetadata = map[string]any{"google": google}
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

// mapGroundingMetadata converts the wire struct to a plain map so it can
// flow through providerMetadata without exposing the internal types.
func mapGroundingMetadata(g *geminiGroundingMetadata) map[string]any {
	out := map[string]any{}
	if len(g.WebSearchQueries) > 0 {
		out["webSearchQueries"] = g.WebSearchQueries
	}
	if len(g.RetrievalQueries) > 0 {
		out["retrievalQueries"] = g.RetrievalQueries
	}
	if g.SearchEntryPoint != nil {
		out["searchEntryPoint"] = map[string]any{"renderedContent": g.SearchEntryPoint.RenderedContent}
	}
	if len(g.GroundingChunks) > 0 {
		chunks := make([]map[string]any, len(g.GroundingChunks))
		for i, c := range g.GroundingChunks {
			entry := map[string]any{}
			if c.Web != nil {
				entry["web"] = map[string]any{"uri": c.Web.URI, "title": c.Web.Title}
			}
			if c.RetrievedContext != nil {
				entry["retrievedContext"] = map[string]any{
					"uri":             c.RetrievedContext.URI,
					"title":           c.RetrievedContext.Title,
					"text":            c.RetrievedContext.Text,
					"fileSearchStore": c.RetrievedContext.FileSearchStore,
				}
			}
			if c.Maps != nil {
				entry["maps"] = map[string]any{
					"uri":     c.Maps.URI,
					"title":   c.Maps.Title,
					"text":    c.Maps.Text,
					"placeId": c.Maps.PlaceID,
				}
			}
			chunks[i] = entry
		}
		out["groundingChunks"] = chunks
	}
	if len(g.GroundingSupports) > 0 {
		supports := make([]map[string]any, len(g.GroundingSupports))
		for i, s := range g.GroundingSupports {
			entry := map[string]any{}
			if s.Segment != nil {
				entry["segment"] = map[string]any{
					"startIndex": s.Segment.StartIndex,
					"endIndex":   s.Segment.EndIndex,
					"text":       s.Segment.Text,
				}
			}
			if len(s.GroundingChunkIndices) > 0 {
				entry["groundingChunkIndices"] = s.GroundingChunkIndices
			}
			if len(s.ConfidenceScores) > 0 {
				entry["confidenceScores"] = s.ConfidenceScores
			}
			supports[i] = entry
		}
		out["groundingSupports"] = supports
	}
	if g.RetrievalMetadata != nil {
		out["retrievalMetadata"] = map[string]any{
			"webDynamicRetrievalScore": g.RetrievalMetadata.WebDynamicRetrievalScore,
		}
	}
	return out
}

func mapURLContextMetadata(u *geminiURLContextMetadata) map[string]any {
	entries := make([]map[string]any, len(u.URLMetadata))
	for i, e := range u.URLMetadata {
		entries[i] = map[string]any{
			"retrievedUrl":       e.RetrievedURL,
			"urlRetrievalStatus": e.URLRetrievalStatus,
		}
	}
	return map[string]any{"urlMetadata": entries}
}

func mapGeminiFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "STOP":
		return stream.FinishReasonStop
	case "MAX_TOKENS":
		return stream.FinishReasonLength
	case "SAFETY", "IMAGE_SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return stream.FinishReasonContentFilter
	case "RECITATION":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonOther
	}
}
