package openaicompat

import (
	"context"
	"fmt"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
)

// EmbeddingModel returns a model for the compatible /embeddings endpoint.
func (p *Provider) EmbeddingModel(id string) model.EmbeddingModel {
	return &CompatEmbeddingModel{id: id, provider: p}
}

type CompatEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *CompatEmbeddingModel) ID() string       { return m.id }
func (m *CompatEmbeddingModel) Provider() string { return m.provider.ID() }
func (m *CompatEmbeddingModel) Dimensions() int  { return 0 }
func (m *CompatEmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.provider.opts.MaxEmbeddingInputs > 0 {
		return m.provider.opts.MaxEmbeddingInputs
	}
	return 2048
}

func (m *CompatEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	if len(opts.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf("too many embedding inputs: %d", len(opts.Values))
	}
	fields := map[string]any{"model": m.id, "input": opts.Values, "encoding_format": "float"}
	if opts.Dimensions != nil {
		fields["dimensions"] = *opts.Dimensions
	}
	for _, key := range []string{"user", "input_type", "truncate", "dimensions"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			fields[key] = value
		}
	}
	var result struct {
		Model string `json:"model"`
		Data  []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	headers, err := m.provider.DoJSON(ctx, "/embeddings", fields, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	if len(result.Data) != len(opts.Values) {
		return nil, fmt.Errorf("%w: embedding count does not match inputs", goaierrors.ErrInvalidResponse)
	}
	out := &model.EmbedResult{Embeddings: make([]model.Embedding, len(result.Data)), Usage: model.EmbeddingUsage{Tokens: result.Usage.TotalTokens}, Response: model.EmbeddingResponse{Model: result.Model, Headers: headers}}
	seen := make(map[int]bool)
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(out.Embeddings) || seen[item.Index] || len(item.Embedding) == 0 {
			return nil, fmt.Errorf("%w: invalid embedding index or vector", goaierrors.ErrInvalidResponse)
		}
		seen[item.Index] = true
		out.Embeddings[item.Index] = model.Embedding{Index: item.Index, Values: item.Embedding}
	}
	return out, nil
}
