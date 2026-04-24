package google

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

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s&alt=sse",
		m.provider.opts.BaseURL, m.id, m.provider.opts.APIKey)

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
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{
			Error: fmt.Errorf("Google AI API error (status %d): %s", resp.StatusCode, string(body)),
		}}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func (m *GoogleModel) buildRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	var warnings []stream.Warning

	// Parse typed provider options
	opts, err := provider.ParseProviderOptions[GenerativeAIOptions](options.ProviderOptions)
	if err != nil {
		return nil, warnings, fmt.Errorf("invalid provider options: %w", err)
	}

	// Unsupported CallOptions on Google (ai-sdk parity).
	if options.FrequencyPenalty != nil {
		warnings = append(warnings, stream.UnsupportedWarning("frequencyPenalty", ""))
	}
	if options.PresencePenalty != nil {
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
			contents = append(contents, geminiContent{
				Role:  "user",
				Parts: convertToGeminiParts(msg.Content),
			})
		case message.RoleAssistant:
			parts := convertAssistantParts(msg.Content)
			contents = append(contents, geminiContent{
				Role:  "model",
				Parts: parts,
			})
		case message.RoleTool:
			// Tool results — emit one functionResponse part per ToolResultPart,
			// then append any attachment parts (ImagePart/FilePart with URL)
			// as sibling inline/fileData parts in the same user turn. This
			// matches ai-sdk's multimodal functionResponse behavior
			// (#47114a3): a remote file URL on a FilePart becomes a fileData
			// part, base64 becomes inlineData, and a URL-bearing ImagePart
			// becomes fileData rather than inlineData.
			var parts []geminiPart
			for _, part := range msg.Content.Parts {
				switch p := part.(type) {
				case message.ToolResultPart:
					parts = append(parts, geminiPart{
						FunctionResponse: &geminiFunctionResponse{
							Name:     p.ToolName,
							Response: map[string]any{"result": p.Result},
						},
					})
				case message.TextPart:
					parts = append(parts, geminiPart{
						FunctionResponse: &geminiFunctionResponse{
							Name:     toolNameFromMessage(msg),
							Response: map[string]any{"result": p.Text},
						},
					})
				case message.ImagePart:
					if strings.HasPrefix(p.Image, "http://") || strings.HasPrefix(p.Image, "https://") {
						parts = append(parts, geminiPart{
							FileData: &geminiFileData{MimeType: p.MimeType, FileURI: p.Image},
						})
					} else {
						parts = append(parts, geminiPart{
							InlineData: &geminiInlineData{MimeType: p.MimeType, Data: p.Image},
						})
					}
					parts = append(parts, geminiPart{
						Text: "Tool executed successfully and returned this image as a response",
					})
				case message.FilePart:
					if p.URL != "" {
						parts = append(parts, geminiPart{
							FileData: &geminiFileData{MimeType: p.MimeType, FileURI: p.URL},
						})
					} else if p.Data != "" {
						parts = append(parts, geminiPart{
							InlineData: &geminiInlineData{MimeType: p.MimeType, Data: p.Data},
						})
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

	// cachedContent
	if opts.CachedContent != "" {
		req.CachedContent = opts.CachedContent
	}

	// thinkingConfig
	if opts.ThinkingConfig != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ThinkingConfig = &geminiThinkingConfig{
			ThinkingBudget:  opts.ThinkingConfig.ThinkingBudget,
			IncludeThoughts: opts.ThinkingConfig.IncludeThoughts,
			ThinkingLevel:   opts.ThinkingConfig.ThinkingLevel,
		}
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

	// serviceTier (ai-sdk #4e22c2c): send as request-root
	// generationConfig.serviceTier. Vertex's mapping to
	// SERVICE_TIER_STANDARD/FLEX/PRIORITY is handled in the Vertex
	// provider wrapper, not here.
	if opts.ServiceTier != "" {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.ServiceTier = opts.ServiceTier
	}

	// streamFunctionCallArguments — only relevant for streaming
	// requests with function tools on Gemini 3+ over Vertex AI.
	// Default is false (ai-sdk #46a3584 flipped the earlier default).
	if opts.StreamFunctionCallArguments != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &geminiGenerationConfig{}
		}
		req.GenerationConfig.StreamFunctionCallArguments = opts.StreamFunctionCallArguments
	}

	body, err := json.Marshal(req)
	return body, warnings, err
}

func (m *GoogleModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	// Convert tools slice to map for name lookup
	toolsByName := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool
	var usage stream.Usage
	var finishReason stream.FinishReason
	var pendingToolCalls []stream.ToolCallEvent
	var groundingMetadata *geminiGroundingMetadata
	var urlContextMetadata *geminiURLContextMetadata

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

		var chunk geminiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
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
					// Text content
					if part.Text != "" {
						if !textStarted {
							textStarted = true
							events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
						}
						events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: part.Text}}
					}

					// Function call
					if part.FunctionCall != nil {
						inputBytes, _ := json.Marshal(part.FunctionCall.Args)
						pendingToolCalls = append(pendingToolCalls, stream.ToolCallEvent{
							ToolCallID: part.FunctionCall.Name, // Gemini uses function name as ID
							ToolName:   part.FunctionCall.Name,
							Input:      inputBytes,
						})
					}
				}
			}
		}

		// Process usage
		if chunk.UsageMetadata != nil {
			usage = stream.UsageFrom(
				chunk.UsageMetadata.PromptTokenCount,
				chunk.UsageMetadata.CandidatesTokenCount,
			)
		}
	}

	// End text if started
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
	if len(pendingToolCalls) > 0 && finishReason == "" {
		finishReason = stream.FinishReasonToolCalls
	}

	// Build provider metadata — surfaces groundingMetadata and
	// urlContextMetadata under providerMetadata.google (mirrors ai-sdk).
	var providerMetadata map[string]any
	if groundingMetadata != nil || urlContextMetadata != nil {
		google := map[string]any{}
		if groundingMetadata != nil {
			google["groundingMetadata"] = mapGroundingMetadata(groundingMetadata)
		}
		if urlContextMetadata != nil {
			google["urlContextMetadata"] = mapURLContextMetadata(urlContextMetadata)
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
	case "SAFETY":
		return stream.FinishReasonContentFilter
	case "RECITATION":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonOther
	}
}
