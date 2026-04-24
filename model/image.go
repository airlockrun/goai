package model

import (
	"context"

	"github.com/airlockrun/goai/stream"
)

// ImageModel is the interface for image generation models.
type ImageModel interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// MaxImagesPerCall returns the maximum number of images that can be generated in a single call.
	MaxImagesPerCall() int

	// Generate generates images based on the provided options.
	Generate(ctx context.Context, opts ImageCallOptions) (*ImageResult, error)
}

// ImageCallOptions contains the options for image generation.
type ImageCallOptions struct {
	// Prompt is the text description of the image to generate.
	Prompt string

	// N is the number of images to generate.
	N int

	// Size is the size of the generated images (e.g., "1024x1024").
	Size string

	// AspectRatio is the aspect ratio (e.g., "16:9", "1:1").
	AspectRatio string

	// Seed for deterministic generation (if supported).
	Seed *int64

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Headers are additional HTTP headers.
	Headers map[string]string

	// Files is an optional list of input images for image-to-image or
	// editing workflows. Each entry is raw bytes in any common image
	// encoding; providers detect the MIME type from the magic bytes
	// before forwarding the payload (ai-sdk ImageModelV3 `files`).
	Files [][]byte

	// Mask is an optional mask image for inpainting where non-zero
	// pixels indicate regions to regenerate (ai-sdk ImageModelV3 `mask`).
	Mask []byte
}

// ImageResult contains the result of an image generation call.
type ImageResult struct {
	// Images contains the generated images.
	Images []GeneratedImage

	// Warnings contains any warnings from the generation process.
	Warnings []stream.Warning

	// Usage contains usage information (if available).
	Usage *ImageUsage

	// Response contains provider-specific response data.
	Response ImageResponse

	// ProviderMetadata contains provider-specific metadata returned by
	// the model (ai-sdk ImageModelV3 `providerMetadata`). Keys are
	// provider IDs, values are provider-defined payloads.
	ProviderMetadata map[string]any
}

// GeneratedImage represents a single generated image.
type GeneratedImage struct {
	// Base64 is the base64-encoded image data.
	Base64 string

	// URL is the URL of the generated image (if available).
	URL string

	// MimeType is the MIME type of the image (e.g., "image/png").
	MimeType string

	// Seed is the seed used for generation (if available).
	Seed *int64

	// RevisedPrompt is the revised prompt used for generation (if available).
	RevisedPrompt string
}

// ImageUsage contains usage information for image generation.
type ImageUsage struct {
	// TotalTokens is the total number of tokens used (for models that use tokens).
	TotalTokens int

	// Steps is the number of diffusion steps (for diffusion models).
	Steps int
}

// ImageResponse contains provider-specific response metadata.
type ImageResponse struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Timestamp is the creation timestamp.
	Timestamp int64

	// Headers contains response headers.
	Headers map[string]string
}
