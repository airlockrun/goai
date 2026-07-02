// Package replicate provides a Replicate provider implementation.
package replicate

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
	defaultBaseURL = "https://api.replicate.com/v1"
)

// Options contains configuration for the Replicate provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Replicate provider.
type Provider struct {
	opts Options
}

// New creates a new Replicate provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                       { return "replicate" }
func (p *Provider) Model(modelID string) stream.Model                { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &ReplicateImageModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)

// ReplicateImageModel implements the ImageModel interface.
type ReplicateImageModel struct {
	id       string
	provider *Provider
}

func (m *ReplicateImageModel) ID() string            { return m.id }
func (m *ReplicateImageModel) Provider() string      { return "replicate" }
func (m *ReplicateImageModel) MaxImagesPerCall() int { return 4 }

func (m *ReplicateImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	// Build input based on model
	input := map[string]any{
		"prompt": opts.Prompt,
	}

	if opts.AspectRatio != "" {
		input["aspect_ratio"] = opts.AspectRatio
	}
	if opts.Seed != nil {
		input["seed"] = *opts.Seed
	}
	if opts.N > 1 {
		input["num_outputs"] = opts.N
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		input[k] = v
	}

	// Create prediction
	reqBody := map[string]any{
		"version": m.id,
		"input":   input,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/predictions", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Token "+m.provider.opts.APIKey)
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

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Replicate API error (status %d): %s", resp.StatusCode, string(body))
	}

	var prediction predictionResponse
	if err := json.Unmarshal(body, &prediction); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for completion
	for prediction.Status == "starting" || prediction.Status == "processing" {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		prediction, err = m.getPrediction(ctx, prediction.ID)
		if err != nil {
			return nil, err
		}
	}

	if prediction.Status == "failed" {
		return nil, fmt.Errorf("prediction failed: %s", prediction.Error)
	}

	// Convert output to images
	var images []model.GeneratedImage
	switch output := prediction.Output.(type) {
	case []any:
		for _, item := range output {
			if url, ok := item.(string); ok {
				images = append(images, model.GeneratedImage{
					URL:      url,
					MimeType: "image/png",
				})
			}
		}
	case string:
		images = append(images, model.GeneratedImage{
			URL:      output,
			MimeType: "image/png",
		})
	}

	return &model.ImageResult{
		Images: images,
		Response: model.ImageResponse{
			ID:    prediction.ID,
			Model: m.id,
		},
	}, nil
}

func (m *ReplicateImageModel) getPrediction(ctx context.Context, id string) (predictionResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", m.provider.opts.BaseURL+"/predictions/"+id, nil)
	if err != nil {
		return predictionResponse{}, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Token "+m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return predictionResponse{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return predictionResponse{}, fmt.Errorf("failed to read response: %w", err)
	}

	var prediction predictionResponse
	if err := json.Unmarshal(body, &prediction); err != nil {
		return predictionResponse{}, fmt.Errorf("failed to parse response: %w", err)
	}

	return prediction, nil
}

type predictionResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Output any    `json:"output"`
	Error  string `json:"error"`
}
