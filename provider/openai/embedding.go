package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// OpenAIEmbeddingModel implements the EmbeddingModel interface for OpenAI.
type OpenAIEmbeddingModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *OpenAIEmbeddingModel) ID() string {
	return m.id
}

// Provider returns "openai".
func (m *OpenAIEmbeddingModel) Provider() string {
	return "openai"
}

// MaxEmbeddingsPerCall returns the maximum number of texts that can be embedded in a single call.
func (m *OpenAIEmbeddingModel) MaxEmbeddingsPerCall() int {
	return 2048 // OpenAI limit
}

// Dimensions returns the default embedding dimensions (0 means variable).
func (m *OpenAIEmbeddingModel) Dimensions() int {
	switch m.id {
	case "text-embedding-3-small":
		return 1536
	case "text-embedding-3-large":
		return 3072
	case "text-embedding-ada-002":
		return 1536
	default:
		return 0
	}
}

// Embed generates embeddings for the provided texts.
func (m *OpenAIEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	// Build request
	req := embeddingRequest{
		Model: m.id,
		Input: opts.Values,
	}

	if opts.Dimensions != nil {
		req.Dimensions = opts.Dimensions
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/embeddings", bytes.NewReader(reqBody))
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
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
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
