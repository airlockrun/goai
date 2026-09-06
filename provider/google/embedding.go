package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// GoogleEmbeddingModel implements the EmbeddingModel interface for Google AI.
type GoogleEmbeddingModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *GoogleEmbeddingModel) ID() string {
	return m.id
}

// Provider returns "google".
func (m *GoogleEmbeddingModel) Provider() string {
	return "google"
}

// MaxEmbeddingsPerCall returns the maximum number of texts that can be embedded in a single call.
func (m *GoogleEmbeddingModel) MaxEmbeddingsPerCall() int {
	return 100
}

// Dimensions returns the default embedding dimensions.
func (m *GoogleEmbeddingModel) Dimensions() int {
	switch m.id {
	case "text-embedding-004":
		return 768
	case "embedding-001":
		return 768
	default:
		return 0
	}
}

// Embed generates embeddings for the provided texts.
func (m *GoogleEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	if len(opts.Values) == 0 || len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf("embedding value count must be between 1 and %d", m.MaxEmbeddingsPerCall())
	}
	// Build request - batch embed
	var content [][]geminiPart
	if value, ok := opts.ProviderOptions["content"]; ok {
		data, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		if len(content) != len(opts.Values) {
			return nil, fmt.Errorf("multimodal content count must match values")
		}
	}
	requests := make([]embedContentRequest, len(opts.Values))
	for i, text := range opts.Values {
		requests[i] = embedContentRequest{
			OutputDimensionality: opts.Dimensions,
			Model:                modelPath(m.id),
			Content: geminiContent{
				Parts: []geminiPart{{Text: text}},
			},
		}
		if value, ok := opts.ProviderOptions["taskType"].(string); ok {
			requests[i].TaskType = value
		}
		if content != nil {
			if text == "" {
				requests[i].Content.Parts = nil
			}
			requests[i].Content.Parts = append(requests[i].Content.Parts, content[i]...)
		}
		if len(opts.Values) > 1 {
			requests[i].Content.Role = "user"
		}
		if value, ok := opts.ProviderOptions["outputDimensionality"]; ok {
			data, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}
			if err := json.Unmarshal(data, &requests[i].OutputDimensionality); err != nil {
				return nil, err
			}
		}
	}

	req := batchEmbedRequest{
		Requests: requests,
	}

	var payload any = req
	action := "batchEmbedContents"
	if len(requests) == 1 {
		payload = requests[0]
		action = "embedContent"
	}
	reqBody, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/%s:%s?key=%s",
		m.provider.opts.BaseURL, modelPath(m.id), action, m.provider.opts.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range m.provider.opts.Headers {
		httpReq.Header.Set(k, v)
	}
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
		return nil, fmt.Errorf("Google AI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var embResp batchEmbedResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	if len(requests) == 1 {
		var single struct {
			Embedding contentEmbedding `json:"embedding"`
		}
		if err := json.Unmarshal(body, &single); err != nil {
			return nil, err
		}
		embResp.Embeddings = []contentEmbedding{single.Embedding}
	}
	if len(embResp.Embeddings) != len(requests) {
		return nil, fmt.Errorf("embedding response count does not match input")
	}
	for _, embedding := range embResp.Embeddings {
		if len(embedding.Values) == 0 {
			return nil, fmt.Errorf("empty embedding response")
		}
	}

	// Convert to model result
	embeddings := make([]model.Embedding, len(embResp.Embeddings))
	for i, e := range embResp.Embeddings {
		embeddings[i] = model.Embedding{
			Values: e.Values,
			Index:  i,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Response: model.EmbeddingResponse{
			Model: m.id,
		},
	}, nil
}

// Request/response types

type embedContentRequest struct {
	OutputDimensionality *int          `json:"outputDimensionality,omitempty"`
	TaskType             string        `json:"taskType,omitempty"`
	Model                string        `json:"model"`
	Content              geminiContent `json:"content"`
}

type batchEmbedRequest struct {
	Requests []embedContentRequest `json:"requests"`
}

type batchEmbedResponse struct {
	Embeddings []contentEmbedding `json:"embeddings"`
}

type contentEmbedding struct {
	Values []float64 `json:"values"`
}
