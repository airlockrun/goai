package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

// CohereEmbeddingModel implements the EmbeddingModel interface for Cohere.
type CohereEmbeddingModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *CohereEmbeddingModel) ID() string {
	return m.id
}

// Provider returns "cohere".
func (m *CohereEmbeddingModel) Provider() string {
	return "cohere"
}

// MaxEmbeddingsPerCall returns the maximum number of texts that can be embedded in a single call.
func (m *CohereEmbeddingModel) MaxEmbeddingsPerCall() int {
	return 96
}

// Dimensions returns the default embedding dimensions.
func (m *CohereEmbeddingModel) Dimensions() int {
	switch m.id {
	case "embed-english-v3.0", "embed-multilingual-v3.0":
		return 1024
	case "embed-english-light-v3.0", "embed-multilingual-light-v3.0":
		return 384
	default:
		return 1024
	}
}

// Embed generates embeddings for the provided texts.
func (m *CohereEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	// Parse provider-specific options (ai-sdk CohereEmbeddingOptions).
	typedOpts, _ := provider.ParseProviderOptions[EmbeddingOptions](opts.ProviderOptions)
	inputType := "search_document"
	if typedOpts != nil && typedOpts.InputType != "" {
		inputType = typedOpts.InputType
	}

	// Build request
	req := embedRequest{
		Model:     m.id,
		Texts:     opts.Values,
		InputType: inputType,
	}
	if typedOpts != nil && typedOpts.OutputDimension > 0 {
		req.OutputDimension = typedOpts.OutputDimension
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/embed", bytes.NewReader(reqBody))
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
		return nil, fmt.Errorf("Cohere API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var embResp embedResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	embeddings := make([]model.Embedding, len(embResp.Embeddings))
	for i, e := range embResp.Embeddings {
		embeddings[i] = model.Embedding{
			Values: e,
			Index:  i,
		}
	}

	tokens := 0
	if embResp.Meta != nil && embResp.Meta.BilledUnits != nil {
		tokens = embResp.Meta.BilledUnits.InputTokens
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: model.EmbeddingUsage{
			Tokens: tokens,
		},
		Response: model.EmbeddingResponse{
			ID: embResp.ID,
		},
	}, nil
}

// Request/response types

type embedRequest struct {
	Model           string   `json:"model"`
	Texts           []string `json:"texts"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	InputType string   `json:"input_type"`
}

type embedResponse struct {
	ID         string      `json:"id"`
	Embeddings [][]float64 `json:"embeddings"`
	Meta       *embedMeta  `json:"meta,omitempty"`
}

type embedMeta struct {
	BilledUnits *embedBilledUnits `json:"billed_units,omitempty"`
}

type embedBilledUnits struct {
	InputTokens int `json:"input_tokens"`
}
