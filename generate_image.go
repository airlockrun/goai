package goai

import (
	"context"
	"fmt"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// ImageInput contains the input for image generation.
type ImageInput struct {
	// Model is the image model to use.
	Model model.ImageModel

	// Prompt is the text description of the image to generate.
	Prompt string

	// N is the number of images to generate (default: 1).
	N int

	// Size is the size of the generated images (e.g., "1024x1024").
	Size string

	// AspectRatio is the aspect ratio (e.g., "16:9", "1:1").
	// Use this instead of Size for models that support aspect ratio.
	AspectRatio string

	// Seed for deterministic generation (if supported).
	Seed *int64

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any
}

// ImageResult contains the result of image generation.
type ImageResult struct {
	// Images contains the generated images.
	Images []GeneratedImage

	// Warnings contains any warnings from the generation process.
	Warnings []stream.Warning

	// Usage contains usage information (if available).
	Usage *ImageUsage

	// Response contains response metadata.
	Response ImageResponseMeta
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

// ImageResponseMeta contains response metadata.
type ImageResponseMeta struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Timestamp is the creation timestamp.
	Timestamp int64

	// Headers contains response headers.
	Headers map[string]string
}

// GenerateImage generates images from an image model.
func GenerateImage(ctx context.Context, input ImageInput) (*ImageResult, error) {
	if input.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if input.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if input.N <= 0 {
		input.N = 1
	}

	// Check if requesting more images than the model supports
	maxImages := input.Model.MaxImagesPerCall()
	if maxImages > 0 && input.N > maxImages {
		return nil, fmt.Errorf("model supports at most %d images per call, got %d", maxImages, input.N)
	}

	// Call the model
	modelResult, err := input.Model.Generate(ctx, model.ImageCallOptions{
		Prompt:          input.Prompt,
		N:               input.N,
		Size:            input.Size,
		AspectRatio:     input.AspectRatio,
		Seed:            input.Seed,
		ProviderOptions: input.ProviderOptions,
		Headers:         input.Headers,
	})
	if err != nil {
		return nil, err
	}

	// Convert model result to goai result
	images := make([]GeneratedImage, len(modelResult.Images))
	for i, img := range modelResult.Images {
		images[i] = GeneratedImage{
			Base64:        img.Base64,
			URL:           img.URL,
			MimeType:      img.MimeType,
			Seed:          img.Seed,
			RevisedPrompt: img.RevisedPrompt,
		}
	}

	var usage *ImageUsage
	if modelResult.Usage != nil {
		usage = &ImageUsage{
			TotalTokens: modelResult.Usage.TotalTokens,
			Steps:       modelResult.Usage.Steps,
		}
	}

	return &ImageResult{
		Images:   images,
		Warnings: modelResult.Warnings,
		Usage:    usage,
		Response: ImageResponseMeta{
			ID:        modelResult.Response.ID,
			Model:     modelResult.Response.Model,
			Timestamp: modelResult.Response.Timestamp,
			Headers:   modelResult.Response.Headers,
		},
	}, nil
}
