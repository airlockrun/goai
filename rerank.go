package goai

import (
	"context"
	"fmt"

	"github.com/airlockrun/goai/model"
)

// RerankInput contains the input for document reranking.
type RerankInput struct {
	// Model is the reranking model to use.
	Model model.RerankingModel

	// Query is the query to rank documents against.
	Query string

	// Documents is the list of documents to rerank.
	Documents []string

	// TopN is the number of top results to return (0 means return all).
	TopN int

	// ReturnDocuments specifies whether to include document text in results.
	ReturnDocuments bool

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any
}

// RerankResult contains the result of document reranking.
type RerankResult struct {
	// Results contains the reranked documents.
	Results []RankedDocument

	// Usage contains usage information.
	Usage RerankUsage

	// Response contains response metadata.
	Response RerankResponseMeta
}

// RankedDocument represents a document with its relevance score.
type RankedDocument struct {
	// Index is the original index of the document.
	Index int

	// Score is the relevance score (higher is more relevant).
	Score float64

	// Document is the document text (if ReturnDocuments was true).
	Document string
}

// RerankUsage contains usage information for reranking.
type RerankUsage struct {
	// SearchUnits is the number of search units used.
	SearchUnits int

	// Tokens is the total number of tokens processed.
	Tokens int
}

// RerankResponseMeta contains response metadata.
type RerankResponseMeta struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for reranking.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}

// Rerank reranks documents based on their relevance to a query.
func Rerank(ctx context.Context, input RerankInput) (*RerankResult, error) {
	if input.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if input.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if len(input.Documents) == 0 {
		return nil, fmt.Errorf("documents are required")
	}

	// Check if requesting more documents than the model supports
	maxDocs := input.Model.MaxDocumentsPerCall()
	if maxDocs > 0 && len(input.Documents) > maxDocs {
		return nil, fmt.Errorf("model supports at most %d documents per call, got %d", maxDocs, len(input.Documents))
	}

	// Call the model
	modelResult, err := input.Model.Rerank(ctx, model.RerankCallOptions{
		Query:           input.Query,
		Documents:       input.Documents,
		TopN:            input.TopN,
		ReturnDocuments: input.ReturnDocuments,
		ProviderOptions: input.ProviderOptions,
		Headers:         input.Headers,
	})
	if err != nil {
		return nil, err
	}

	// Convert model result to goai result
	results := make([]RankedDocument, len(modelResult.Results))
	for i, r := range modelResult.Results {
		results[i] = RankedDocument{
			Index:    r.Index,
			Score:    r.Score,
			Document: r.Document,
		}
	}

	return &RerankResult{
		Results: results,
		Usage: RerankUsage{
			SearchUnits: modelResult.Usage.SearchUnits,
			Tokens:      modelResult.Usage.Tokens,
		},
		Response: RerankResponseMeta{
			ID:      modelResult.Response.ID,
			Model:   modelResult.Response.Model,
			Headers: modelResult.Response.Headers,
		},
	}, nil
}
