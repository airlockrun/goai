package testutil

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// MockEmbeddingModel is a mock implementation of EmbeddingModel for testing.
type MockEmbeddingModel struct {
	id                   string
	provider             string
	dimensions           int
	maxEmbeddingsPerCall int
	DoEmbedFunc          func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error)
	DoEmbedCalls         []model.EmbedCallOptions
	EmbedResponse        *model.EmbedResult
}

// MockEmbeddingModelOptions configures the mock embedding model.
type MockEmbeddingModelOptions struct {
	ID                   string
	Provider             string
	Dimensions           int
	MaxEmbeddingsPerCall int
	DoEmbedFunc          func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error)
	EmbedResponse        *model.EmbedResult
}

// NewMockEmbeddingModel creates a new mock embedding model.
func NewMockEmbeddingModel(opts MockEmbeddingModelOptions) *MockEmbeddingModel {
	m := &MockEmbeddingModel{
		id:                   opts.ID,
		provider:             opts.Provider,
		dimensions:           opts.Dimensions,
		maxEmbeddingsPerCall: opts.MaxEmbeddingsPerCall,
		DoEmbedFunc:          opts.DoEmbedFunc,
		EmbedResponse:        opts.EmbedResponse,
	}
	if m.id == "" {
		m.id = "mock-model-id"
	}
	if m.provider == "" {
		m.provider = "mock-provider"
	}
	if m.maxEmbeddingsPerCall == 0 {
		m.maxEmbeddingsPerCall = 100
	}
	return m
}

func (m *MockEmbeddingModel) ID() string                { return m.id }
func (m *MockEmbeddingModel) Provider() string          { return m.provider }
func (m *MockEmbeddingModel) Dimensions() int           { return m.dimensions }
func (m *MockEmbeddingModel) MaxEmbeddingsPerCall() int { return m.maxEmbeddingsPerCall }

func (m *MockEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	m.DoEmbedCalls = append(m.DoEmbedCalls, opts)

	if m.DoEmbedFunc != nil {
		return m.DoEmbedFunc(ctx, opts)
	}

	if m.EmbedResponse != nil {
		return m.EmbedResponse, nil
	}

	return &model.EmbedResult{
		Embeddings: []model.Embedding{
			{Values: []float64{0.1, 0.2, 0.3}, Index: 0},
		},
		Usage: model.EmbeddingUsage{Tokens: 10},
	}, nil
}
