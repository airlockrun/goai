package google

// GenerativeAIOptions contains provider-specific options for the Google Generative AI API.
// These options match ai-sdk's GoogleGenerativeAIProviderOptions schema.
// See: ai-sdk/packages/google/src/google-generative-ai-options.ts
type GenerativeAIOptions struct {
	// ResponseModalities controls the response format.
	// Values: "TEXT", "IMAGE"
	ResponseModalities []string `json:"responseModalities,omitempty"`

	// ThinkingConfig configures the thinking/reasoning behavior.
	ThinkingConfig *ThinkingConfig `json:"thinkingConfig,omitempty"`

	// CachedContent is the name of cached content used as context.
	// Format: cachedContents/{cachedContent}
	CachedContent string `json:"cachedContent,omitempty"`

	// StructuredOutputs enables structured output. Default is true.
	// Useful when JSON Schema contains unsupported elements.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// SafetySettings is a list of unique safety settings for blocking unsafe content.
	SafetySettings []SafetySetting `json:"safetySettings,omitempty"`

	// Threshold is a global safety threshold setting.
	// Values: "HARM_BLOCK_THRESHOLD_UNSPECIFIED", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE", "OFF"
	Threshold string `json:"threshold,omitempty"`

	// AudioTimestamp enables timestamp understanding for audio-only files.
	AudioTimestamp *bool `json:"audioTimestamp,omitempty"`

	// Labels defines labels used in billing reports. Available on Vertex AI only.
	Labels map[string]string `json:"labels,omitempty"`

	// MediaResolution specifies the media resolution.
	// Values: "MEDIA_RESOLUTION_UNSPECIFIED", "MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"
	MediaResolution string `json:"mediaResolution,omitempty"`

	// ImageConfig configures the image generation aspect ratio for Gemini models.
	ImageConfig *ImageConfig `json:"imageConfig,omitempty"`

	// RetrievalConfig provides location context for Google Maps and Google Search grounding.
	RetrievalConfig *RetrievalConfig `json:"retrievalConfig,omitempty"`

	// StreamFunctionCallArguments enables streaming of function call
	// arguments as they're produced. Vertex AI only, Gemini 3+ only.
	// Default is false (ai-sdk #46a3584 flipped the default).
	StreamFunctionCallArguments *bool `json:"streamFunctionCallArguments,omitempty"`

	// ServiceTier selects the service tier. Values: "standard", "flex",
	// "priority" (ai-sdk #4e22c2c). On Vertex, goai maps these to the
	// SERVICE_TIER_* wire values.
	ServiceTier string `json:"serviceTier,omitempty"`
}

// ThinkingConfig configures Google's thinking/reasoning behavior.
type ThinkingConfig struct {
	// ThinkingBudget is the token budget for thinking.
	ThinkingBudget int `json:"thinkingBudget,omitempty"`

	// IncludeThoughts controls whether to include thoughts in output.
	IncludeThoughts *bool `json:"includeThoughts,omitempty"`

	// ThinkingLevel controls the thinking level.
	// Values: "minimal", "low", "medium", "high"
	ThinkingLevel string `json:"thinkingLevel,omitempty"`
}

// SafetySetting represents a safety setting for content blocking.
type SafetySetting struct {
	// Category of harm to block.
	// Values: "HARM_CATEGORY_UNSPECIFIED", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT",
	//         "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_CIVIC_INTEGRITY"
	Category string `json:"category"`

	// Threshold for blocking.
	// Values: "HARM_BLOCK_THRESHOLD_UNSPECIFIED", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
	//         "BLOCK_ONLY_HIGH", "BLOCK_NONE", "OFF"
	Threshold string `json:"threshold"`
}

// ImageConfig configures image generation.
type ImageConfig struct {
	// AspectRatio for generated images.
	// Values: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
	AspectRatio string `json:"aspectRatio,omitempty"`

	// ImageSize for generated images.
	// Values: "1K", "2K", "4K"
	ImageSize string `json:"imageSize,omitempty"`
}

// RetrievalConfig provides location context for grounding.
type RetrievalConfig struct {
	// LatLng is the latitude/longitude for location context.
	LatLng *LatLng `json:"latLng,omitempty"`
}

// LatLng represents a geographic coordinate.
type LatLng struct {
	// Latitude of the location.
	Latitude float64 `json:"latitude"`

	// Longitude of the location.
	Longitude float64 `json:"longitude"`
}
