package togetherai

import (
	"context"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
)

type TogetherRerankingModel struct {
	id       string
	provider *Provider
}

func (m *TogetherRerankingModel) ID() string               { return m.id }
func (m *TogetherRerankingModel) Provider() string         { return "togetherai" }
func (m *TogetherRerankingModel) MaxDocumentsPerCall() int { return 1000 }
func (m *TogetherRerankingModel) Rerank(ctx context.Context, opts model.RerankCallOptions) (*model.RerankResult, error) {
	if len(opts.Documents) > m.MaxDocumentsPerCall() {
		return nil, errors.New("too many Together reranking documents")
	}
	body := map[string]any{"model": m.id, "query": opts.Query, "documents": opts.Documents, "return_documents": opts.ReturnDocuments}
	if opts.TopN > 0 {
		body["top_n"] = opts.TopN
	}
	var result struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"results"`
	}
	headers, err := m.provider.compat.DoJSON(ctx, "/rerank", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	out := &model.RerankResult{Response: model.RerankResponse{ID: result.ID, Model: result.Model, Headers: headers}}
	for _, item := range result.Results {
		if item.Index < 0 || item.Index >= len(opts.Documents) {
			return nil, fmt.Errorf("invalid Together reranking index %d", item.Index)
		}
		doc := model.RankedDocument{Index: item.Index, Score: item.Score}
		if opts.ReturnDocuments {
			doc.Document = opts.Documents[item.Index]
		}
		out.Results = append(out.Results, doc)
	}
	return out, nil
}
