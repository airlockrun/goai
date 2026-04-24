package model

import (
	"context"

	"github.com/airlockrun/goai/stream"
)

// RerankingModel is the interface for document reranking models.
type RerankingModel interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// MaxDocumentsPerCall returns the maximum number of documents that can be reranked in a single call.
	MaxDocumentsPerCall() int

	// Rerank reranks documents based on their relevance to a query.
	Rerank(ctx context.Context, opts RerankCallOptions) (*RerankResult, error)
}

// RerankCallOptions contains the options for reranking.
type RerankCallOptions struct {
	// Query is the query to rank documents against.
	Query string

	// Documents is the list of documents to rerank.
	Documents []string

	// TopN is the number of top results to return (0 means return all).
	TopN int

	// ReturnDocuments specifies whether to include document text in results.
	ReturnDocuments bool

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Headers are additional HTTP headers.
	Headers map[string]string
}

// RerankResult contains the result of a reranking call.
type RerankResult struct {
	// Results contains the reranked documents.
	Results []RankedDocument

	// Usage contains usage information.
	Usage RerankUsage

	// Warnings contains any warnings from the reranking process.
	// Mirrors ai-sdk's RerankingModelV3 `warnings: SharedV3Warning[]`.
	Warnings []stream.Warning

	// Response contains provider-specific response data.
	Response RerankResponse
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

// RerankResponse contains provider-specific response metadata.
type RerankResponse struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for reranking.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}
