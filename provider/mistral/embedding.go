package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// MistralEmbeddingModel implements the EmbeddingModel interface for Mistral.
type MistralEmbeddingModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *MistralEmbeddingModel) ID() string {
	return m.id
}

// Provider returns "mistral".
func (m *MistralEmbeddingModel) Provider() string {
	return "mistral"
}

// MaxEmbeddingsPerCall returns the maximum number of texts that can be embedded in a single call.
func (m *MistralEmbeddingModel) MaxEmbeddingsPerCall() int {
	return 16384
}

// Dimensions returns the default embedding dimensions.
func (m *MistralEmbeddingModel) Dimensions() int {
	return 1024
}

// Embed generates embeddings for the provided texts.
func (m *MistralEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	// Build request
	req := embeddingRequest{
		Model: m.id,
		Input: opts.Values,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	baseURL := m.provider.opts.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/embeddings", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("Mistral API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	embeddings := make([]model.Embedding, len(embResp.Data))
	for i, d := range embResp.Data {
		embeddings[i] = model.Embedding{
			Values: d.Embedding,
			Index:  d.Index,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: model.EmbeddingUsage{
			Tokens: embResp.Usage.TotalTokens,
		},
		Response: model.EmbeddingResponse{
			Model: embResp.Model,
		},
	}, nil
}

// Request/response types

type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Data   []embeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingUsage  `json:"usage"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
