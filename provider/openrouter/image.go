package openrouter

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

// imageModel implements model.ImageModel against OpenRouter's dedicated
// POST {base}/images endpoint (note: NOT /images/generations like OpenAI).
// The response is OpenAI-compatible: {created, data:[{b64_json|url}]}.
type imageModel struct {
	id       string
	provider *Provider
}

func (m *imageModel) ID() string            { return m.id }
func (m *imageModel) Provider() string      { return "openrouter" }
func (m *imageModel) MaxImagesPerCall() int { return 1 }

func (m *imageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	var warnings []stream.Warning
	if opts.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}

	req := imageRequest{Model: m.id, Prompt: opts.Prompt, N: opts.N, Size: opts.Size}
	if req.N <= 0 {
		req.N = 1
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/images", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.provider.setHeaders(httpReq.Header, opts.Headers)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenRouter API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var imgResp imageResponse
	if err := json.Unmarshal(respBody, &imgResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

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
		Response: model.ImageResponse{Model: m.id, Timestamp: imgResp.Created},
	}, nil
}

type imageRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	N      int    `json:"n,omitempty"`
	Size   string `json:"size,omitempty"`
}

type imageResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		URL           string `json:"url,omitempty"`
		B64JSON       string `json:"b64_json,omitempty"`
		RevisedPrompt string `json:"revised_prompt,omitempty"`
	} `json:"data"`
}
