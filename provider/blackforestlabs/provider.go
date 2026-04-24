// Package blackforestlabs provides a Black Forest Labs (Flux) provider implementation.
package blackforestlabs

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
	defaultBaseURL = "https://api.bfl.ml/v1"
)

// Options contains configuration for the Black Forest Labs provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Black Forest Labs provider.
type Provider struct {
	opts Options
}

// New creates a new Black Forest Labs provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                       { return "black-forest-labs" }
func (p *Provider) Model(modelID string) stream.Model                { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &FluxImageModel{
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
		"flux-pro-1.1",
		"flux-pro",
		"flux-dev",
		"flux-pro-1.1-ultra",
	}
}

var _ provider.Provider = (*Provider)(nil)

// FluxImageModel implements the ImageModel interface.
type FluxImageModel struct {
	id       string
	provider *Provider
}

func (m *FluxImageModel) ID() string            { return m.id }
func (m *FluxImageModel) Provider() string      { return "black-forest-labs" }
func (m *FluxImageModel) MaxImagesPerCall() int { return 1 }

func (m *FluxImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	reqBody := map[string]any{
		"prompt": opts.Prompt,
	}

	// Set dimensions
	width := 1024
	height := 1024
	if opts.Size != "" {
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
	} else if opts.AspectRatio != "" {
		switch opts.AspectRatio {
		case "16:9":
			width, height = 1344, 768
		case "9:16":
			width, height = 768, 1344
		case "21:9":
			width, height = 1536, 640
		case "9:21":
			width, height = 640, 1536
		case "4:3":
			width, height = 1152, 896
		case "3:4":
			width, height = 896, 1152
		}
	}
	reqBody["width"] = width
	reqBody["height"] = height

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

	// Create task
	taskURL := fmt.Sprintf("%s/%s", m.provider.opts.BaseURL, m.id)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", taskURL, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Key", m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("Black Forest Labs API error (status %d): %s", resp.StatusCode, string(body))
	}

	var taskResp taskResponse
	if err := json.Unmarshal(body, &taskResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for result
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}

		result, status, err := m.getTaskResult(ctx, taskResp.ID)
		if err != nil {
			return nil, err
		}

		switch status {
		case "Ready":
			return &model.ImageResult{
				Images: []model.GeneratedImage{
					{
						URL:      result.Sample,
						MimeType: "image/jpeg",
					},
				},
				Response: model.ImageResponse{
					ID:    taskResp.ID,
					Model: m.id,
				},
			}, nil
		case "Error":
			return nil, fmt.Errorf("task failed")
		case "Pending", "Processing":
			// Continue polling
		}
	}
}

func (m *FluxImageModel) getTaskResult(ctx context.Context, taskID string) (*taskResult, string, error) {
	url := fmt.Sprintf("%s/get_result?id=%s", m.provider.opts.BaseURL, taskID)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("X-Key", m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read response: %w", err)
	}

	var result taskResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, "", fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, result.Status, nil
}

type taskResponse struct {
	ID string `json:"id"`
}

type taskResult struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Sample string `json:"sample"` // URL to the image
}
