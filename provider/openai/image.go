package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// gptImageFamilyPrefixes matches model IDs that share the gpt-image
// surface (auto size default, aspectRatio→size mapping, b64_json
// response by default — so response_format is omitted on the wire).
// Mirrors ai-sdk's OpenAIImageModelId list and hasDefaultResponseFormat
// (packages/openai/src/image/openai-image-options.ts).
var gptImageFamilyPrefixes = []string{
	"chatgpt-image-",
	"gpt-image-1-mini",
	"gpt-image-1.5",
	"gpt-image-1",
	"gpt-image-2",
}

func isGPTImageFamily(id string) bool {
	for _, p := range gptImageFamilyPrefixes {
		if strings.HasPrefix(id, p) {
			return true
		}
	}
	return false
}

// OpenAIImageModel implements the ImageModel interface for OpenAI.
type OpenAIImageModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *OpenAIImageModel) ID() string {
	return m.id
}

// Provider returns "openai".
func (m *OpenAIImageModel) Provider() string {
	return "openai"
}

// MaxImagesPerCall returns the maximum number of images that can be generated in a single call.
func (m *OpenAIImageModel) MaxImagesPerCall() int {
	switch m.id {
	case "dall-e-3":
		return 1 // DALL-E 3 only supports 1 image per call
	case "dall-e-2":
		return 10
	}
	if isGPTImageFamily(m.id) {
		return 10
	}
	return 1
}

// Generate generates images based on the provided options.
func (m *OpenAIImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	// Mirrors ai-sdk openai-image-model.ts: the gpt-image family can map
	// aspectRatio into size; all other models ignore both aspectRatio and
	// seed.
	gptImage := isGPTImageFamily(m.id)
	var warnings []stream.Warning
	if opts.AspectRatio != "" && !gptImage {
		warnings = append(warnings, stream.UnsupportedWarning("aspectRatio", "This model does not support aspect ratio. Use `size` instead."))
	}
	if opts.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}

	// Build request
	req := imageRequest{
		Model:  m.id,
		Prompt: opts.Prompt,
		N:      opts.N,
	}

	if req.N <= 0 {
		req.N = 1
	}

	// Set size
	if opts.Size != "" {
		req.Size = opts.Size
	} else {
		// Default sizes based on model
		switch {
		case m.id == "dall-e-3", m.id == "dall-e-2":
			req.Size = "1024x1024"
		case gptImage:
			req.Size = "auto"
		}
	}

	// Aspect ratio maps to size for the gpt-image family.
	if opts.AspectRatio != "" && gptImage {
		req.Size = opts.AspectRatio
	}

	// Mirrors ai-sdk's hasDefaultResponseFormat: the gpt-image family
	// returns b64_json by default and rejects an explicit response_format
	// on some endpoints, so omit it for those models.
	if !gptImage {
		req.ResponseFormat = "b64_json"
	}

	// Extract OpenAI-specific provider options
	openaiOpts, _ := opts.ProviderOptions["openai"].(map[string]any)

	// Quality setting for DALL-E 3
	if m.id == "dall-e-3" {
		if quality, ok := openaiOpts["quality"].(string); ok {
			req.Quality = quality
		} else {
			req.Quality = "standard"
		}

		if style, ok := openaiOpts["style"].(string); ok {
			req.Style = style
		}
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/images/generations", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	if m.provider.opts.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", m.provider.opts.Organization)
	}
	if m.provider.opts.Project != "" {
		httpReq.Header.Set("OpenAI-Project", m.provider.opts.Project)
	}
	// Provider-level headers
	for k, v := range m.provider.opts.Headers {
		httpReq.Header.Set(k, v)
	}
	// Request-level headers (override provider headers)
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
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var imgResp imageResponse
	if err := json.Unmarshal(body, &imgResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	images := make([]model.GeneratedImage, len(imgResp.Data))
	for i, d := range imgResp.Data {
		images[i] = model.GeneratedImage{
			Base64:        d.B64JSON,
			URL:           d.URL,
			MimeType:      "image/png",
			RevisedPrompt: d.RevisedPrompt,
		}
	}

	return &model.ImageResult{
		Images:   images,
		Warnings: warnings,
		Response: model.ImageResponse{
			Model:     m.id,
			Timestamp: imgResp.Created,
		},
	}, nil
}

// Request/response types

type imageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Quality        string `json:"quality,omitempty"`
	Style          string `json:"style,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

type imageResponse struct {
	Created int64       `json:"created"`
	Data    []imageData `json:"data"`
}

type imageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}
