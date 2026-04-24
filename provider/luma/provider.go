// Package luma provides a Luma Labs provider implementation for video/image generation.
package luma

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
	defaultBaseURL = "https://api.lumalabs.ai/dream-machine/v1"
)

// Options contains configuration for the Luma Labs provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Luma Labs provider.
type Provider struct {
	opts Options
}

// New creates a new Luma Labs provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                       { return "luma" }
func (p *Provider) Model(modelID string) stream.Model                { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &LumaImageModel{
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
		"photon-1",
		"photon-flash-1",
	}
}

var _ provider.Provider = (*Provider)(nil)

// LumaImageModel implements the ImageModel interface.
type LumaImageModel struct {
	id       string
	provider *Provider
}

func (m *LumaImageModel) ID() string            { return m.id }
func (m *LumaImageModel) Provider() string      { return "luma" }
func (m *LumaImageModel) MaxImagesPerCall() int { return 1 }

func (m *LumaImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	reqBody := map[string]any{
		"prompt": opts.Prompt,
		"model":  m.id,
	}

	// Aspect ratio
	if opts.AspectRatio != "" {
		reqBody["aspect_ratio"] = opts.AspectRatio
	} else if opts.Size != "" {
		// Convert size to aspect ratio
		var width, height int
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
		if width == height {
			reqBody["aspect_ratio"] = "1:1"
		} else if width > height {
			reqBody["aspect_ratio"] = "16:9"
		} else {
			reqBody["aspect_ratio"] = "9:16"
		}
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/generations/image", m.provider.opts.BaseURL)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
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

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("Luma API error (status %d): %s", resp.StatusCode, string(body))
	}

	var genResp struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &genResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for result
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		result, err := m.getResult(ctx, genResp.ID)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
	}
}

func (m *LumaImageModel) getResult(ctx context.Context, generationID string) (*model.ImageResult, error) {
	url := fmt.Sprintf("%s/generations/%s", m.provider.opts.BaseURL, generationID)

	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)

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
		ID     string `json:"id"`
		State  string `json:"state"`
		Assets struct {
			Image string `json:"image"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(body, &statusResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	switch statusResp.State {
	case "completed":
		return &model.ImageResult{
			Images: []model.GeneratedImage{
				{
					URL:      statusResp.Assets.Image,
					MimeType: "image/png",
				},
			},
			Response: model.ImageResponse{
				ID:    generationID,
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
