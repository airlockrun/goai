package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// AzureImageModel implements the ImageModel interface.
type AzureImageModel struct {
	id       string
	provider *Provider
}

func (m *AzureImageModel) ID() string            { return m.id }
func (m *AzureImageModel) Provider() string      { return "azure" }
func (m *AzureImageModel) MaxImagesPerCall() int { return 1 }

func (m *AzureImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	reqBody := map[string]any{
		"prompt": opts.Prompt,
		"n":      1,
	}

	if opts.N > 0 {
		reqBody["n"] = opts.N
	}

	// Size mapping
	if opts.Size != "" {
		reqBody["size"] = opts.Size
	} else if opts.AspectRatio != "" {
		// Map aspect ratio to size
		switch opts.AspectRatio {
		case "1:1":
			reqBody["size"] = "1024x1024"
		case "16:9":
			reqBody["size"] = "1792x1024"
		case "9:16":
			reqBody["size"] = "1024x1792"
		default:
			reqBody["size"] = "1024x1024"
		}
	}

	// Provider-specific options
	if style, ok := opts.ProviderOptions["style"].(string); ok {
		reqBody["style"] = style
	}
	if quality, ok := opts.ProviderOptions["quality"].(string); ok {
		reqBody["quality"] = quality
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/images/generations?api-version=%s",
		m.provider.baseURL(m.id), m.provider.opts.APIVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Azure API error (status %d): %s", resp.StatusCode, string(body))
	}

	var imgResp imageResponse
	if err := json.Unmarshal(body, &imgResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	images := make([]model.GeneratedImage, len(imgResp.Data))
	for i, item := range imgResp.Data {
		images[i] = model.GeneratedImage{
			URL:           item.URL,
			Base64:        item.B64JSON,
			MimeType:      "image/png",
			RevisedPrompt: item.RevisedPrompt,
		}
	}

	return &model.ImageResult{
		Images: images,
		Response: model.ImageResponse{
			ID:    fmt.Sprintf("%d", imgResp.Created),
			Model: m.id,
		},
	}, nil
}

type imageResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		URL           string `json:"url"`
		B64JSON       string `json:"b64_json"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
}
