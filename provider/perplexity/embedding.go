package perplexity

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

type EmbeddingOptions struct {
	EncodingFormat string `json:"encodingFormat,omitempty"`
	Dimensions     *int   `json:"dimensions,omitempty"`
}
type PerplexityEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *PerplexityEmbeddingModel) ID() string                { return m.id }
func (m *PerplexityEmbeddingModel) Provider() string          { return "perplexity" }
func (m *PerplexityEmbeddingModel) Dimensions() int           { return 0 }
func (m *PerplexityEmbeddingModel) MaxEmbeddingsPerCall() int { return 512 }
func (m *PerplexityEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	if len(opts.Values) > 512 {
		return nil, errors.New("Perplexity accepts at most 512 embedding inputs")
	}
	settings, err := provider.ParseProviderOptions[EmbeddingOptions](opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	format := settings.EncodingFormat
	if format == "" {
		format = "base64_int8"
	}
	if format != "base64_int8" && format != "base64_binary" {
		return nil, fmt.Errorf("unsupported Perplexity encoding format %q", format)
	}
	body := map[string]any{"model": m.id, "input": opts.Values, "encoding_format": format}
	dimensions := settings.Dimensions
	if dimensions == nil {
		dimensions = opts.Dimensions
	}
	if dimensions != nil {
		body["dimensions"] = *dimensions
	}
	var result struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	headers, err := m.provider.compat.DoJSON(ctx, "/v1/embeddings", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	if len(result.Data) != len(opts.Values) {
		return nil, errors.New("Perplexity embedding count does not match inputs")
	}
	out := &model.EmbedResult{Usage: model.EmbeddingUsage{Tokens: result.Usage.PromptTokens}, Response: model.EmbeddingResponse{Model: m.id, Headers: headers}}
	for i, item := range result.Data {
		data, err := base64.StdEncoding.DecodeString(item.Embedding)
		if err != nil {
			return nil, err
		}
		values := make([]float64, len(data))
		for j, value := range data {
			if format == "base64_int8" {
				values[j] = float64(int8(value))
			} else {
				values[j] = float64(value)
			}
		}
		out.Embeddings = append(out.Embeddings, model.Embedding{Index: i, Values: values})
	}
	return out, nil
}
