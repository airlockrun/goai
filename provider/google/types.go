package google

import (
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/message"
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
type geminiToolConfig struct {
	FunctionCallingConfig *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
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
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// geminiFileData references a remotely-hosted file by URI. Mirrors the
// fileData part in ai-sdk's convert-to-google-generative-ai-messages.ts;
// used when a FilePart or ImagePart carries a URL instead of base64 data.
type geminiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type geminiFunctionResponse struct {
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
	WebSearchQueries  []string                   `json:"webSearchQueries,omitempty"`
	RetrievalQueries  []string                   `json:"retrievalQueries,omitempty"`
	SearchEntryPoint  *geminiSearchEntryPoint    `json:"searchEntryPoint,omitempty"`
	GroundingChunks   []geminiGroundingChunk     `json:"groundingChunks,omitempty"`
	GroundingSupports []geminiGroundingSupport   `json:"groundingSupports,omitempty"`
	RetrievalMetadata *geminiRetrievalMetadata   `json:"retrievalMetadata,omitempty"`
}

type geminiSearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitempty"`
}

type geminiGroundingChunk struct {
	Web              *geminiGroundingWeb        `json:"web,omitempty"`
	RetrievedContext *geminiRetrievedContext    `json:"retrievedContext,omitempty"`
	Maps             *geminiGroundingMapsChunk  `json:"maps,omitempty"`
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
	RetrievedURL        string `json:"retrievedUrl,omitempty"`
	URLRetrievalStatus  string `json:"urlRetrievalStatus,omitempty"`
}

type geminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
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

func convertToGeminiParts(content message.Content) []geminiPart {
	// If simple text, return single text part
	if content.Text != "" && len(content.Parts) == 0 {
		return []geminiPart{{Text: content.Text}}
	}

	result := make([]geminiPart, 0, len(content.Parts))
	for _, part := range content.Parts {
		switch p := part.(type) {
		case message.TextPart:
			result = append(result, geminiPart{Text: p.Text})
		case message.ImagePart:
			if strings.HasPrefix(p.Image, "http://") || strings.HasPrefix(p.Image, "https://") {
				result = append(result, geminiPart{
					FileData: &geminiFileData{MimeType: p.MimeType, FileURI: p.Image},
				})
			} else {
				result = append(result, geminiPart{
					InlineData: &geminiInlineData{MimeType: p.MimeType, Data: p.Image},
				})
			}
		case message.FilePart:
			if p.URL != "" {
				result = append(result, geminiPart{
					FileData: &geminiFileData{MimeType: p.MimeType, FileURI: p.URL},
				})
			} else if p.Data != "" {
				result = append(result, geminiPart{
					InlineData: &geminiInlineData{MimeType: p.MimeType, Data: p.Data},
				})
			}
		}
	}
	return result
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
