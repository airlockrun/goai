package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// GoogleImageModel implements the ImageModel interface for Google AI.
type GoogleImageModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *GoogleImageModel) ID() string {
	return m.id
}

// Provider returns "google".
func (m *GoogleImageModel) Provider() string {
	return "google"
}

// MaxImagesPerCall returns the maximum number of images that can be generated in a single call.
func (m *GoogleImageModel) MaxImagesPerCall() int {
	return 4
}

// Generate generates images based on the provided options.
func (m *GoogleImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	// Mirrors ai-sdk google-generative-ai-image-model.ts: Imagen does not
	// expose a `size` or `seed` parameter, so both are silently dropped
	// with a warning.
	var warnings []stream.Warning
	if opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}
	if opts.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", "This model does not support the `seed` option through this provider."))
	}

	// Build request using Imagen API
	req := imagenRequest{
		Instances: []imagenInstance{{
			Prompt: opts.Prompt,
		}},
		Parameters: imagenParameters{
			SampleCount: opts.N,
		},
	}

	if req.Parameters.SampleCount <= 0 {
		req.Parameters.SampleCount = 1
	}

	// Handle aspect ratio
	if opts.AspectRatio != "" {
		req.Parameters.AspectRatio = opts.AspectRatio
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/models/%s:predict?key=%s",
		m.provider.opts.BaseURL, m.id, m.provider.opts.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	// Execute request
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Google AI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var imgResp imagenResponse
	if err := json.Unmarshal(body, &imgResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	images := make([]model.GeneratedImage, len(imgResp.Predictions))
	for i, p := range imgResp.Predictions {
		images[i] = model.GeneratedImage{
			Base64:   p.BytesBase64Encoded,
			MimeType: p.MimeType,
		}
		if p.MimeType == "" {
			images[i].MimeType = "image/png"
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

// Request/response types

type imagenRequest struct {
	Instances  []imagenInstance `json:"instances"`
	Parameters imagenParameters `json:"parameters"`
}

type imagenInstance struct {
	Prompt string `json:"prompt"`
}

type imagenParameters struct {
	SampleCount int    `json:"sampleCount,omitempty"`
	AspectRatio string `json:"aspectRatio,omitempty"`
}

type imagenResponse struct {
	Predictions []imagenPrediction `json:"predictions"`
}

type imagenPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}
