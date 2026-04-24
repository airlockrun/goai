package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// CohereRerankingModel implements the RerankingModel interface for Cohere.
type CohereRerankingModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *CohereRerankingModel) ID() string {
	return m.id
}

// Provider returns "cohere".
func (m *CohereRerankingModel) Provider() string {
	return "cohere"
}

// MaxDocumentsPerCall returns the maximum number of documents that can be reranked in a single call.
func (m *CohereRerankingModel) MaxDocumentsPerCall() int {
	return 1000
}

// Rerank reranks documents based on their relevance to a query.
func (m *CohereRerankingModel) Rerank(ctx context.Context, opts model.RerankCallOptions) (*model.RerankResult, error) {
	// Build request
	req := rerankRequest{
		Model:     m.id,
		Query:     opts.Query,
		Documents: opts.Documents,
	}

	if opts.TopN > 0 {
		req.TopN = opts.TopN
	}
	if opts.ReturnDocuments {
		req.ReturnDocuments = true
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/rerank", bytes.NewReader(reqBody))
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
	var rerankResp rerankResponse
	if err := json.Unmarshal(body, &rerankResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	results := make([]model.RankedDocument, len(rerankResp.Results))
	for i, r := range rerankResp.Results {
		results[i] = model.RankedDocument{
			Index:    r.Index,
			Score:    r.RelevanceScore,
			Document: r.Document.Text,
		}
	}

	searchUnits := 0
	if rerankResp.Meta != nil && rerankResp.Meta.BilledUnits != nil {
		searchUnits = rerankResp.Meta.BilledUnits.SearchUnits
	}

	return &model.RerankResult{
		Results: results,
		Usage: model.RerankUsage{
			SearchUnits: searchUnits,
		},
		Response: model.RerankResponse{
			ID:    rerankResp.ID,
			Model: m.id,
		},
	}, nil
}

// Request/response types

type rerankRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            int      `json:"top_n,omitempty"`
	ReturnDocuments bool     `json:"return_documents,omitempty"`
}

type rerankResponse struct {
	ID      string         `json:"id"`
	Results []rerankResult `json:"results"`
	Meta    *rerankMeta    `json:"meta,omitempty"`
}

type rerankResult struct {
	Index          int            `json:"index"`
	RelevanceScore float64        `json:"relevance_score"`
	Document       rerankDocument `json:"document,omitempty"`
}

type rerankDocument struct {
	Text string `json:"text"`
}

type rerankMeta struct {
	BilledUnits *rerankBilledUnits `json:"billed_units,omitempty"`
}

type rerankBilledUnits struct {
	SearchUnits int `json:"search_units"`
}
