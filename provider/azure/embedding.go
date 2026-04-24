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

// AzureEmbeddingModel implements the EmbeddingModel interface.
type AzureEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *AzureEmbeddingModel) ID() string                { return m.id }
func (m *AzureEmbeddingModel) Provider() string          { return "azure" }
func (m *AzureEmbeddingModel) MaxEmbeddingsPerCall() int { return 2048 }
func (m *AzureEmbeddingModel) Dimensions() int           { return 0 } // Variable dimensions

func (m *AzureEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	reqBody := map[string]any{
		"input": opts.Values,
	}

	if opts.Dimensions != nil {
		reqBody["dimensions"] = *opts.Dimensions
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/embeddings?api-version=%s",
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

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	embeddings := make([]model.Embedding, len(embResp.Data))
	for i, item := range embResp.Data {
		embeddings[i] = model.Embedding{
			Values: item.Embedding,
			Index:  item.Index,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: model.EmbeddingUsage{
			Tokens: embResp.Usage.TotalTokens,
		},
		Response: model.EmbeddingResponse{
			Model: m.id,
		},
	}, nil
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}
