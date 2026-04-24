// Package prodia provides a Prodia provider implementation for image generation.
package prodia

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
	defaultBaseURL = "https://api.prodia.com/v1"
)

// Options contains configuration for the Prodia provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Prodia provider.
type Provider struct {
	opts Options
}

// New creates a new Prodia provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                       { return "prodia" }
func (p *Provider) Model(modelID string) stream.Model                { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &ProdiaImageModel{
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
		"sdxl",
		"sd_xl_base_1.0.safetensors [be9edd61]",
		"dreamshaper_8.safetensors [9d40847d]",
		"absolutereality_v181.safetensors [3d9d4d2b]",
		"realistic_vision_v5.safetensors [614d1063]",
	}
}

var _ provider.Provider = (*Provider)(nil)

// ProdiaImageModel implements the ImageModel interface.
type ProdiaImageModel struct {
	id       string
	provider *Provider
}

func (m *ProdiaImageModel) ID() string            { return m.id }
func (m *ProdiaImageModel) Provider() string      { return "prodia" }
func (m *ProdiaImageModel) MaxImagesPerCall() int { return 1 }

func (m *ProdiaImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	// Determine endpoint based on model
	endpoint := "/sd/generate"
	if m.id == "sdxl" || m.id == "sd_xl_base_1.0.safetensors [be9edd61]" {
		endpoint = "/sdxl/generate"
	}

	reqBody := map[string]any{
		"prompt": opts.Prompt,
	}

	if m.id != "sdxl" {
		reqBody["model"] = m.id
	}

	// Size
	if opts.Size != "" {
		var width, height int
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
		reqBody["width"] = width
		reqBody["height"] = height
	} else if opts.AspectRatio != "" {
		switch opts.AspectRatio {
		case "1:1":
			reqBody["width"] = 1024
			reqBody["height"] = 1024
		case "16:9":
			reqBody["width"] = 1024
			reqBody["height"] = 576
		case "9:16":
			reqBody["width"] = 576
			reqBody["height"] = 1024
		}
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

	url := m.provider.opts.BaseURL + endpoint

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Prodia-Key", m.provider.opts.APIKey)
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

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Prodia API error (status %d): %s", resp.StatusCode, string(body))
	}

	var jobResp struct {
		Job string `json:"job"`
	}
	if err := json.Unmarshal(body, &jobResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for result
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		result, err := m.getResult(ctx, jobResp.Job)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
	}
}

func (m *ProdiaImageModel) getResult(ctx context.Context, jobID string) (*model.ImageResult, error) {
	url := fmt.Sprintf("%s/job/%s", m.provider.opts.BaseURL, jobID)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("X-Prodia-Key", m.provider.opts.APIKey)

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
		Job      string `json:"job"`
		Status   string `json:"status"`
		ImageURL string `json:"imageUrl"`
	}
	if err := json.Unmarshal(body, &statusResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	switch statusResp.Status {
	case "succeeded":
		return &model.ImageResult{
			Images: []model.GeneratedImage{
				{
					URL:      statusResp.ImageURL,
					MimeType: "image/png",
				},
			},
			Response: model.ImageResponse{
				ID:    jobID,
				Model: m.id,
			},
		}, nil
	case "failed":
		return nil, fmt.Errorf("image generation failed")
	default:
		// Still processing
		return nil, nil
	}
}
