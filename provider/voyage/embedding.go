package voyage

import (
	"context"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

type EmbeddingOptions struct {
	InputType       string `json:"inputType,omitempty"`
	Truncation      *bool  `json:"truncation,omitempty"`
	OutputDimension *int   `json:"outputDimension,omitempty"`
	OutputDtype     string `json:"outputDtype,omitempty"`
}
type EmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *EmbeddingModel) ID() string                { return m.id }
func (m *EmbeddingModel) Provider() string          { return "voyage" }
func (m *EmbeddingModel) Dimensions() int           { return 0 }
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int { return 128 }
func (m *EmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	if len(opts.Values) > 128 {
		return nil, errors.New("Voyage accepts at most 128 embedding inputs")
	}
	settings, err := provider.ParseProviderOptions[EmbeddingOptions](opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if settings.InputType != "" && settings.InputType != "query" && settings.InputType != "document" {
		return nil, fmt.Errorf("invalid Voyage input type %q", settings.InputType)
	}
	switch settings.OutputDtype {
	case "", "float", "int8", "uint8", "binary", "ubinary":
	default:
		return nil, fmt.Errorf("invalid Voyage output dtype %q", settings.OutputDtype)
	}
	body := map[string]any{"model": m.id, "input": opts.Values}
	if settings.InputType != "" {
		body["input_type"] = settings.InputType
	}
	if settings.Truncation != nil {
		body["truncation"] = *settings.Truncation
	}
	if settings.OutputDtype != "" {
		body["output_dtype"] = settings.OutputDtype
	}
	dimensions := settings.OutputDimension
	if dimensions == nil {
		dimensions = opts.Dimensions
	}
	if dimensions != nil {
		if *dimensions <= 0 {
			return nil, errors.New("output dimension must be positive")
		}
		body["output_dimension"] = *dimensions
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
	headers, err := m.provider.client.DoJSON(ctx, "/embeddings", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	if len(result.Data) != len(opts.Values) {
		return nil, errors.New("Voyage embedding count does not match inputs")
	}
	out := &model.EmbedResult{Embeddings: make([]model.Embedding, len(result.Data)), Usage: model.EmbeddingUsage{Tokens: result.Usage.TotalTokens}, Response: model.EmbeddingResponse{Model: result.Model, Headers: headers}}
	seen := map[int]bool{}
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(out.Embeddings) || seen[item.Index] {
			return nil, fmt.Errorf("invalid Voyage embedding index %d", item.Index)
		}
		seen[item.Index] = true
		out.Embeddings[item.Index] = model.Embedding{Index: item.Index, Values: item.Embedding}
	}
	return out, nil
}
