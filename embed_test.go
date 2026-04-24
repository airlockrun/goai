package goai

import (
	"context"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/testutil"
)

// Translated from ai-sdk/packages/ai/src/embed/embed.test.ts

var dummyEmbedding = []float64{0.1, 0.2, 0.3}
var testEmbedValue = "sunny day at the beach"

func TestEmbed_ResultEmbedding(t *testing.T) {
	t.Run("should generate embedding", func(t *testing.T) {
		mockModel := testutil.NewMockEmbeddingModel(testutil.MockEmbeddingModelOptions{
			DoEmbedFunc: func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
				if len(opts.Values) != 1 || opts.Values[0] != testEmbedValue {
					t.Errorf("expected values %v, got %v", []string{testEmbedValue}, opts.Values)
				}
				return &model.EmbedResult{
					Embeddings: []model.Embedding{
						{Values: dummyEmbedding, Index: 0},
					},
					Usage: model.EmbeddingUsage{Tokens: 10},
				}, nil
			},
		})

		result, err := Embed(context.Background(), EmbedInput{
			Model: mockModel,
			Value: testEmbedValue,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embedding) != len(dummyEmbedding) {
			t.Errorf("expected embedding length %d, got %d", len(dummyEmbedding), len(result.Embedding))
		}

		for i, v := range result.Embedding {
			if v != dummyEmbedding[i] {
				t.Errorf("expected embedding[%d] = %f, got %f", i, dummyEmbedding[i], v)
			}
		}
	})
}

func TestEmbed_ResultUsage(t *testing.T) {
	t.Run("should include usage in the result", func(t *testing.T) {
		mockModel := testutil.NewMockEmbeddingModel(testutil.MockEmbeddingModelOptions{
			DoEmbedFunc: func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
				return &model.EmbedResult{
					Embeddings: []model.Embedding{
						{Values: dummyEmbedding, Index: 0},
					},
					Usage: model.EmbeddingUsage{Tokens: 10},
				}, nil
			},
		})

		result, err := Embed(context.Background(), EmbedInput{
			Model: mockModel,
			Value: testEmbedValue,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Usage.Tokens != 10 {
			t.Errorf("expected usage tokens 10, got %d", result.Usage.Tokens)
		}
	})
}

func TestEmbed_OptionsHeaders(t *testing.T) {
	t.Run("should set headers", func(t *testing.T) {
		mockModel := testutil.NewMockEmbeddingModel(testutil.MockEmbeddingModelOptions{
			DoEmbedFunc: func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
				if opts.Headers == nil {
					t.Error("expected headers to be set")
				}
				if opts.Headers["custom-request-header"] != "request-header-value" {
					t.Errorf("expected custom header, got %v", opts.Headers)
				}
				return &model.EmbedResult{
					Embeddings: []model.Embedding{
						{Values: dummyEmbedding, Index: 0},
					},
				}, nil
			},
		})

		result, err := Embed(context.Background(), EmbedInput{
			Model: mockModel,
			Value: testEmbedValue,
			Headers: map[string]string{
				"custom-request-header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		for i, v := range result.Embedding {
			if v != dummyEmbedding[i] {
				t.Errorf("expected embedding[%d] = %f, got %f", i, dummyEmbedding[i], v)
			}
		}
	})
}

func TestEmbed_OptionsProviderOptions(t *testing.T) {
	t.Run("should pass provider options to model", func(t *testing.T) {
		mockModel := testutil.NewMockEmbeddingModel(testutil.MockEmbeddingModelOptions{
			DoEmbedFunc: func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
				if opts.ProviderOptions == nil {
					t.Error("expected provider options to be set")
				}
				if opts.ProviderOptions["aProvider"] == nil {
					t.Error("expected aProvider in provider options")
				}
				providerOpts, ok := opts.ProviderOptions["aProvider"].(map[string]any)
				if !ok {
					t.Error("expected aProvider to be a map")
				}
				if providerOpts["someKey"] != "someValue" {
					t.Errorf("expected someKey = someValue, got %v", providerOpts["someKey"])
				}
				return &model.EmbedResult{
					Embeddings: []model.Embedding{
						{Values: []float64{1, 2, 3}, Index: 0},
					},
				}, nil
			},
		})

		result, err := Embed(context.Background(), EmbedInput{
			Model: mockModel,
			Value: "test-input",
			ProviderOptions: map[string]any{
				"aProvider": map[string]any{"someKey": "someValue"},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expected := []float64{1, 2, 3}
		for i, v := range result.Embedding {
			if v != expected[i] {
				t.Errorf("expected embedding[%d] = %f, got %f", i, expected[i], v)
			}
		}
	})
}

func TestEmbedMany_ShouldGenerateMultipleEmbeddings(t *testing.T) {
	values := []string{"sunny day at the beach", "rainy day in the city"}
	embeddings := [][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5},
		{0.6, 0.7, 0.8, 0.9, 1.0},
	}

	mockModel := testutil.NewMockEmbeddingModel(testutil.MockEmbeddingModelOptions{
		DoEmbedFunc: func(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
			if len(opts.Values) != 2 {
				t.Errorf("expected 2 values, got %d", len(opts.Values))
			}
			return &model.EmbedResult{
				Embeddings: []model.Embedding{
					{Values: embeddings[0], Index: 0},
					{Values: embeddings[1], Index: 1},
				},
				Usage: model.EmbeddingUsage{Tokens: 20},
			}, nil
		},
	})

	result, err := EmbedMany(context.Background(), EmbedInput{
		Model:  mockModel,
		Values: values,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 2 {
		t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
	}

	for i, emb := range result.Embeddings {
		for j, v := range emb.Values {
			if v != embeddings[i][j] {
				t.Errorf("expected embedding[%d][%d] = %f, got %f", i, j, embeddings[i][j], v)
			}
		}
	}
}
