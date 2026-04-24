package xai

// XaiImageOptions contains provider-specific options for the xAI image
// generation endpoint. Mirrors ai-sdk's xaiImageModelOptions schema.
// See: ai-sdk/packages/xai/src/xai-image-options.ts
type XaiImageOptions struct {
	// AspectRatio overrides the call-level aspect ratio when no top-level
	// value is supplied (e.g. "16:9", "1:1").
	AspectRatio string `json:"aspect_ratio,omitempty"`

	// OutputFormat selects the desired image encoding (e.g. "jpeg", "png").
	OutputFormat string `json:"output_format,omitempty"`

	// SyncMode requests synchronous generation when true.
	SyncMode *bool `json:"sync_mode,omitempty"`

	// Resolution selects the output resolution for grok-imagine models.
	// Values: "1k", "2k".
	Resolution string `json:"resolution,omitempty"`

	// Quality selects the rendering quality tier.
	// Values: "low", "medium", "high".
	Quality string `json:"quality,omitempty"`

	// User is an optional end-user identifier forwarded to xAI for abuse
	// monitoring.
	User string `json:"user,omitempty"`
}
