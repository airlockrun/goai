package voyage

import (
	"context"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

type RerankingOptions struct {
	ReturnDocuments *bool `json:"returnDocuments,omitempty"`
	Truncation      *bool `json:"truncation,omitempty"`
}
type RerankingModel struct {
	id       string
	provider *Provider
}

func (m *RerankingModel) ID() string               { return m.id }
func (m *RerankingModel) Provider() string         { return "voyage" }
func (m *RerankingModel) MaxDocumentsPerCall() int { return 1000 }
func (m *RerankingModel) Rerank(ctx context.Context, opts model.RerankCallOptions) (*model.RerankResult, error) {
	if len(opts.Documents) > m.MaxDocumentsPerCall() {
		return nil, errors.New("too many Voyage reranking documents")
	}
	settings, err := provider.ParseProviderOptions[RerankingOptions](opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	returnDocuments := opts.ReturnDocuments
	if settings.ReturnDocuments != nil {
		returnDocuments = *settings.ReturnDocuments
	}
	body := map[string]any{"model": m.id, "query": opts.Query, "documents": opts.Documents, "return_documents": returnDocuments}
	if opts.TopN > 0 {
		body["top_k"] = opts.TopN
	}
	if settings.Truncation != nil {
		body["truncation"] = *settings.Truncation
	}
	var result struct {
		Data []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"data"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	headers, err := m.provider.client.DoJSON(ctx, "/rerank", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	out := &model.RerankResult{Response: model.RerankResponse{Model: m.id, Headers: headers}, Usage: model.RerankUsage{Tokens: result.Usage.TotalTokens}}
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(opts.Documents) {
			return nil, fmt.Errorf("invalid Voyage reranking index %d", item.Index)
		}
		doc := model.RankedDocument{Index: item.Index, Score: item.Score}
		if returnDocuments {
			doc.Document = opts.Documents[item.Index]
		}
		out.Results = append(out.Results, doc)
	}
	return out, nil
}
