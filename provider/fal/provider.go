// Package fal provides a fal.ai provider implementation for image generation.
package fal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://queue.fal.run"
)

// Options contains configuration for the fal provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the fal provider.
type Provider struct {
	opts Options
}

// New creates a new fal provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                       { return "fal" }
func (p *Provider) Model(modelID string) stream.Model                { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &FalImageModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		"fal-ai/flux-pro/v1.1",
		"fal-ai/flux/dev",
		"fal-ai/flux/schnell",
		"fal-ai/flux-lora",
		"fal-ai/stable-diffusion-v3-medium",
		"fal-ai/aura-flow",
	}
}

var _ provider.Provider = (*Provider)(nil)

// FalImageModel implements the ImageModel interface.
type FalImageModel struct {
	id       string
	provider *Provider
}

func (m *FalImageModel) ID() string            { return m.id }
func (m *FalImageModel) Provider() string      { return "fal" }
func (m *FalImageModel) MaxImagesPerCall() int { return 4 }

func (m *FalImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	reqBody := map[string]any{
		"prompt": opts.Prompt,
	}

	// Number of images
	if opts.N > 0 {
		reqBody["num_images"] = opts.N
	}

	// Size
	if opts.Size != "" {
		var width, height int
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
		reqBody["image_size"] = map[string]int{
			"width":  width,
			"height": height,
		}
	} else if opts.AspectRatio != "" {
		// fal supports aspect ratios directly
		reqBody["image_size"] = opts.AspectRatio
	}

	// Seed
	if opts.Seed != nil {
		reqBody["seed"] = *opts.Seed
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Submit to queue
	url := fmt.Sprintf("%s/%s", m.provider.opts.BaseURL, m.id)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Key "+m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return nil, fmt.Errorf("fal API error (status %d): %s", resp.StatusCode, string(body))
	}

	var queueResp queueResponse
	if err := json.Unmarshal(body, &queueResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for result
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}

		result, err := m.getResult(ctx, queueResp.RequestID)
		if err != nil {
			return nil, err
		}

		if result != nil {
			return result, nil
		}
	}
}

func (m *FalImageModel) getResult(ctx context.Context, requestID string) (*model.ImageResult, error) {
	url := fmt.Sprintf("%s/%s/requests/%s/status", m.provider.opts.BaseURL, m.id, requestID)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Key "+m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var statusResp struct {
		Status   string `json:"status"`
		Response struct {
			Images []struct {
				URL         string `json:"url"`
				ContentType string `json:"content_type"`
			} `json:"images"`
		} `json:"response"`
	}
	if err := json.Unmarshal(body, &statusResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	switch statusResp.Status {
	case "COMPLETED":
		images := make([]model.GeneratedImage, len(statusResp.Response.Images))
		for i, img := range statusResp.Response.Images {
			mimeType := img.ContentType
			if mimeType == "" {
				mimeType = "image/jpeg"
			}
			images[i] = model.GeneratedImage{
				URL:      img.URL,
				MimeType: mimeType,
			}
		}
		return &model.ImageResult{
			Images: images,
			Response: model.ImageResponse{
				ID:    requestID,
				Model: m.id,
			},
		}, nil
	case "FAILED":
		return nil, fmt.Errorf("image generation failed")
	default:
		// Still processing
		return nil, nil
	}
}

type queueResponse struct {
	RequestID string `json:"request_id"`
	Status    string `json:"status"`
}
