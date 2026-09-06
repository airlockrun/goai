// Package deepinfra provides a DeepInfra provider implementation.
package deepinfra

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.deepinfra.com/v1/openai"
)

// Options contains configuration for the DeepInfra provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the DeepInfra provider.
type Provider struct {
	compat *openaicompat.Provider
	opts   Options
}

// New creates a new DeepInfra provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{
		opts: opts,
		compat: openaicompat.New(openaicompat.Options{
			ProviderID: "deepinfra",
			APIKey:     opts.APIKey,
			BaseURL:    opts.BaseURL,
			Headers:    opts.Headers,
		}),
	}
}

func (p *Provider) ID() string { return "deepinfra" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.compat.Model(modelID)
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &DeepInfraEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &DeepInfraImageModel{id: modelID, provider: p}
}
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

var _ provider.Provider = (*Provider)(nil)

// DeepInfraEmbeddingModel implements the EmbeddingModel interface.
type DeepInfraEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *DeepInfraEmbeddingModel) ID() string                { return m.id }
func (m *DeepInfraEmbeddingModel) Provider() string          { return "deepinfra" }
func (m *DeepInfraEmbeddingModel) MaxEmbeddingsPerCall() int { return 100 }
func (m *DeepInfraEmbeddingModel) Dimensions() int           { return 0 }

func (m *DeepInfraEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	reqBody := map[string]any{
		"model": m.id,
		"input": opts.Values,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := m.provider.opts.BaseURL + "/embeddings"

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("DeepInfra API error (status %d): %s", resp.StatusCode, string(body))
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
