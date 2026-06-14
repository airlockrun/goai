package google

import (
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Request types

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"`
	SafetySettings    []geminiSafetySetting   `json:"safetySettings,omitempty"`
	CachedContent     string                  `json:"cachedContent,omitempty"`
}

// geminiToolConfig configures tool calling. Mirrors Gemini's toolConfig wire
// shape (ai-sdk: google-prepare-tools.ts). functionCallingConfig.mode is one
// of "AUTO" / "ANY" / "NONE"; allowedFunctionNames restricts ANY-mode calls
// to a specific subset.
//
// includeServerSideToolInvocations enables Gemini to surface invocations of
// server-side tools (e.g. google_search, code_execution) in the response
// stream. Set on direct Gemini only — the Vertex endpoint rejects this
// field (ai-sdk PR #14767). goai's vertex package is implemented separately
// and does not share this struct.
type geminiToolConfig struct {
	FunctionCallingConfig            *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
	IncludeServerSideToolInvocations bool                         `json:"includeServerSideToolInvocations,omitempty"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// geminiSafetySetting represents a safety setting for content blocking.
type geminiSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	// ThoughtSignature carries the opaque thinking-state signature for
	// Gemini 3 thinking models. Surfaced via providerMetadata.google.
	// Vertex's multi-turn rule: if a turn has any function calls, at
	// least one must carry a thoughtSignature. ai-sdk #14968.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// geminiFileData references a remotely-hosted file by URI. Mirrors the
// fileData part in ai-sdk's convert-to-google-generative-ai-messages.ts;
// used when a FilePart carries a URL.
type geminiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type geminiFunctionCall struct {
	// ID is the call identifier when the Gemini API returns one. goai
	// uses it as the ToolCallID on inbound calls and echoes it back as the
	// matching functionResponse.id on outbound results. ai-sdk #15317.
	ID   string `json:"id,omitempty"`
	Name string `json:"name"`
	// Args may be omitted on no-args calls. Vertex emits `{name: "X"}`
	// without args/partialArgs/willContinue for zero-arg tools; ai-sdk
	// #14968. We default to "{}" on the wire.
	Args map[string]any `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	// ID echoes the originating functionCall.id so Gemini can correlate the
	// result with the call. ai-sdk #15317.
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature                 *float64              `json:"temperature,omitempty"`
	TopP                        *float64              `json:"topP,omitempty"`
	TopK                        *int                  `json:"topK,omitempty"`
	MaxOutputTokens             *int                  `json:"maxOutputTokens,omitempty"`
	StopSequences               []string              `json:"stopSequences,omitempty"`
	ResponseModalities          []string              `json:"responseModalities,omitempty"`
	ResponseMimeType            string                `json:"responseMimeType,omitempty"`
	ResponseSchema              json.RawMessage       `json:"responseSchema,omitempty"`
	ThinkingConfig              *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
	AudioTimestamp              *bool                 `json:"audioTimestamp,omitempty"`
	MediaResolution             string                `json:"mediaResolution,omitempty"`
	ServiceTier                 string                `json:"serviceTier,omitempty"`
	StreamFunctionCallArguments *bool                 `json:"streamFunctionCallArguments,omitempty"`
}

// geminiThinkingConfig configures thinking/reasoning behavior.
type geminiThinkingConfig struct {
	ThinkingBudget  int    `json:"thinkingBudget,omitempty"`
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
}

// geminiTool is a single entry in the Gemini request's tools[] array.
// Only one field is populated per entry:
//   - FunctionDeclarations: user-defined function tools.
//   - GoogleSearch / GoogleSearchRetrieval / GoogleMaps / EnterpriseWebSearch /
//     URLContext / CodeExecution: provider-defined grounding tools.
type geminiTool struct {
	FunctionDeclarations  []geminiFunctionDeclaration  `json:"functionDeclarations,omitempty"`
	GoogleSearch          *geminiGoogleSearch          `json:"googleSearch,omitempty"`
	GoogleSearchRetrieval *geminiGoogleSearchRetrieval `json:"googleSearchRetrieval,omitempty"`
	GoogleMaps            *struct{}                    `json:"googleMaps,omitempty"`
	EnterpriseWebSearch   *struct{}                    `json:"enterpriseWebSearch,omitempty"`
	URLContext            *struct{}                    `json:"urlContext,omitempty"`
	CodeExecution         *struct{}                    `json:"codeExecution,omitempty"`
}

// geminiGoogleSearch is the Gemini 2+ googleSearch payload. Empty by
// default; opt into searchTypes / timeRangeFilter via the corresponding
// GoogleSearchOptions fields (ai-sdk #2565e70).
type geminiGoogleSearch struct {
	SearchTypes     *geminiGoogleSearchTypes     `json:"searchTypes,omitempty"`
	TimeRangeFilter *geminiGoogleSearchTimeRange `json:"timeRangeFilter,omitempty"`
}

type geminiGoogleSearchTypes struct {
	WebSearch   *struct{} `json:"webSearch,omitempty"`
	ImageSearch *struct{} `json:"imageSearch,omitempty"`
}

type geminiGoogleSearchTimeRange struct {
	StartTime string `json:"startTime"`
	EndTime   string `json:"endTime"`
}

// geminiGoogleSearchRetrieval is the older (Gemini 1.5) shape with dynamic
// retrieval config. Gemini 2+ uses the plain {googleSearch:{}} shape.
type geminiGoogleSearchRetrieval struct {
	DynamicRetrievalConfig *geminiDynamicRetrievalConfig `json:"dynamicRetrievalConfig,omitempty"`
}

type geminiDynamicRetrievalConfig struct {
	Mode             string  `json:"mode,omitempty"`
	DynamicThreshold float64 `json:"dynamicThreshold,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Response types

type geminiStreamChunk struct {
	Candidates    []geminiCandidate    `json:"candidates,omitempty"`
	UsageMetadata *geminiUsageMetadata `json:"usageMetadata,omitempty"`
}

type geminiCandidate struct {
	Content            *geminiContent            `json:"content,omitempty"`
	FinishReason       string                    `json:"finishReason,omitempty"`
	Index              int                       `json:"index,omitempty"`
	GroundingMetadata  *geminiGroundingMetadata  `json:"groundingMetadata,omitempty"`
	URLContextMetadata *geminiURLContextMetadata `json:"urlContextMetadata,omitempty"`
}

// geminiGroundingMetadata mirrors Gemini's groundingMetadata response field.
// Surfaced via providerMetadata.google.groundingMetadata.
type geminiGroundingMetadata struct {
	WebSearchQueries  []string                 `json:"webSearchQueries,omitempty"`
	RetrievalQueries  []string                 `json:"retrievalQueries,omitempty"`
	SearchEntryPoint  *geminiSearchEntryPoint  `json:"searchEntryPoint,omitempty"`
	GroundingChunks   []geminiGroundingChunk   `json:"groundingChunks,omitempty"`
	GroundingSupports []geminiGroundingSupport `json:"groundingSupports,omitempty"`
	RetrievalMetadata *geminiRetrievalMetadata `json:"retrievalMetadata,omitempty"`
}

type geminiSearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitempty"`
}

type geminiGroundingChunk struct {
	Web              *geminiGroundingWeb       `json:"web,omitempty"`
	RetrievedContext *geminiRetrievedContext   `json:"retrievedContext,omitempty"`
	Maps             *geminiGroundingMapsChunk `json:"maps,omitempty"`
}

type geminiGroundingWeb struct {
	URI   string `json:"uri,omitempty"`
	Title string `json:"title,omitempty"`
}

type geminiRetrievedContext struct {
	URI             string `json:"uri,omitempty"`
	Title           string `json:"title,omitempty"`
	Text            string `json:"text,omitempty"`
	FileSearchStore string `json:"fileSearchStore,omitempty"`
}

type geminiGroundingMapsChunk struct {
	URI     string `json:"uri,omitempty"`
	Title   string `json:"title,omitempty"`
	Text    string `json:"text,omitempty"`
	PlaceID string `json:"placeId,omitempty"`
}

type geminiGroundingSupport struct {
	Segment               *geminiGroundingSegment `json:"segment,omitempty"`
	GroundingChunkIndices []int                   `json:"groundingChunkIndices,omitempty"`
	ConfidenceScores      []float64               `json:"confidenceScores,omitempty"`
}

type geminiGroundingSegment struct {
	StartIndex int    `json:"startIndex,omitempty"`
	EndIndex   int    `json:"endIndex,omitempty"`
	Text       string `json:"text,omitempty"`
}

type geminiRetrievalMetadata struct {
	WebDynamicRetrievalScore float64 `json:"webDynamicRetrievalScore,omitempty"`
}

// geminiURLContextMetadata surfaces urlContextMetadata.urlMetadata from the
// response.
type geminiURLContextMetadata struct {
	URLMetadata []geminiURLMetadataEntry `json:"urlMetadata,omitempty"`
}

type geminiURLMetadataEntry struct {
	RetrievedURL       string `json:"retrievedUrl,omitempty"`
	URLRetrievalStatus string `json:"urlRetrievalStatus,omitempty"`
}

type geminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	// CachedContentTokenCount is the cached portion of the prompt; thoughts
	// (reasoning) tokens are reported separately from candidates.
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	// ServiceTier is the tier that served the request. Surfaced via
	// providerMetadata.google.serviceTier. ai-sdk #15488.
	ServiceTier string `json:"serviceTier,omitempty"`
}

// toUsage maps Gemini's usageMetadata to stream.Usage, splitting the cached
// prompt portion onto InputTokens.{CacheRead,NoCache} and the thoughts portion
// onto OutputTokens.Reasoning. The output total is candidates + thoughts, since
// Gemini reports candidatesTokenCount exclusive of reasoning. Mirrors ai-sdk's
// convert-google-usage.ts.
func (u geminiUsageMetadata) toUsage() stream.Usage {
	cached := u.CachedContentTokenCount
	thoughts := u.ThoughtsTokenCount

	out := stream.Usage{
		InputTokens:  stream.InputTokens{Total: stream.IntPtr(u.PromptTokenCount)},
		OutputTokens: stream.OutputTokens{Total: stream.IntPtr(u.CandidatesTokenCount + thoughts)},
	}
	if cached > 0 {
		out.InputTokens.CacheRead = stream.IntPtr(cached)
		out.InputTokens.NoCache = stream.IntPtr(u.PromptTokenCount - cached)
	} else {
		out.InputTokens.NoCache = stream.IntPtr(u.PromptTokenCount)
	}
	if thoughts > 0 {
		out.OutputTokens.Reasoning = stream.IntPtr(thoughts)
	}
	out.OutputTokens.Text = stream.IntPtr(u.CandidatesTokenCount)
	return out
}

// Conversion functions

func toolNameFromMessage(msg message.Message) string {
	for _, part := range msg.Content.Parts {
		if tr, ok := part.(message.ToolResultPart); ok {
			return tr.ToolName
		}
	}
	return ""
}

func toolCallIDFromMessage(msg message.Message) string {
	for _, part := range msg.Content.Parts {
		if tr, ok := part.(message.ToolResultPart); ok {
			return tr.ToolCallID
		}
	}
	return ""
}

func getTextFromContent(content message.Content) string {
	if content.Text != "" {
		return content.Text
	}
	for _, part := range content.Parts {
		if tp, ok := part.(message.TextPart); ok {
			return tp.Text
		}
	}
	return ""
}

// isImageMimeType reports whether a FilePart carries an image, matching the
// "image/" prefix convention the message package uses to fold images into
// FilePart.
func isImageMimeType(mimeType string) bool {
	return strings.HasPrefix(mimeType, "image/")
}

// convertToGeminiParts maps message content parts to Gemini wire parts.
// Inline base64 FileParts become inlineData; URL-backed FileParts become
// fileData. Reference and text file data have no Gemini request mapping and
// surface as unsupported warnings.
func convertToGeminiParts(content message.Content) ([]geminiPart, []stream.Warning) {
	// If simple text, return single text part
	if content.Text != "" && len(content.Parts) == 0 {
		return []geminiPart{{Text: content.Text}}, nil
	}

	var warnings []stream.Warning
	result := make([]geminiPart, 0, len(content.Parts))
	for _, part := range content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			result = append(result, geminiPart{Text: p.Text})
		case message.FilePart:
			switch d := p.Data.(type) {
			case message.FileDataURL:
				result = append(result, geminiPart{
					FileData: &geminiFileData{MimeType: p.MimeType, FileURI: d.URL},
				})
			case message.FileDataBytes:
				result = append(result, geminiPart{
					InlineData: &geminiInlineData{MimeType: p.MimeType, Data: d.Data},
				})
			case message.FileDataText, message.FileDataReference:
				warnings = append(warnings, stream.UnsupportedWarning("filePart", "file data type not supported by Gemini"))
			}
		}
	}
	return result, warnings
}

func convertAssistantParts(content message.Content) []geminiPart {
	result := make([]geminiPart, 0)

	// Add text if present
	text := getTextFromContent(content)
	if text != "" {
		result = append(result, geminiPart{Text: text})
	}

	// Add tool calls as function calls
	for _, part := range content.Parts {
		if tc, ok := part.(message.ToolCallPart); ok {
			var args map[string]any
			json.Unmarshal(tc.Input, &args)
			result = append(result, geminiPart{
				FunctionCall: &geminiFunctionCall{
					ID:   tc.ID,
					Name: tc.Name,
					Args: args,
				},
			})
		}
	}

	return result
}

func convertToGeminiFunctions(tools []tool.Tool) []geminiFunctionDeclaration {
	result := make([]geminiFunctionDeclaration, 0, len(tools))

	for _, t := range tools {
		result = append(result, geminiFunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}

	return result
}
