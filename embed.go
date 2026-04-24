package goai

import (
	"context"
	"fmt"

	"github.com/airlockrun/goai/model"
)

// EmbedInput contains the input for embedding generation.
type EmbedInput struct {
	// Model is the embedding model to use.
	Model model.EmbeddingModel

	// Value is the text to embed (for single embedding).
	Value string

	// Values is a list of texts to embed (for batch embedding).
	// If Value is set, it takes precedence.
	Values []string

	// Dimensions is the desired embedding dimensions (if model supports it).
	Dimensions *int

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any
}

// EmbedResult contains the result of embedding generation.
type EmbedResult struct {
	// Embedding is the generated embedding (for single value input).
	Embedding []float64

	// Embeddings contains all generated embeddings (for batch input).
	Embeddings []Embedding

	// Usage contains usage information.
	Usage EmbeddingUsage

	// Response contains response metadata.
	Response EmbeddingResponseMeta
}

// Embedding represents a single embedding.
type Embedding struct {
	// Values is the embedding vector.
	Values []float64

	// Index is the index of the input text this embedding corresponds to.
	Index int
}

// EmbeddingUsage contains usage information for embedding generation.
type EmbeddingUsage struct {
	// Tokens is the total number of tokens used.
	Tokens int
}

// EmbeddingResponseMeta contains response metadata.
type EmbeddingResponseMeta struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}

// Embed generates an embedding for a single text.
func Embed(ctx context.Context, input EmbedInput) (*EmbedResult, error) {
	if input.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if input.Value == "" && len(input.Values) == 0 {
		return nil, fmt.Errorf("value or values is required")
	}

	// Use single value if provided
	values := input.Values
	if input.Value != "" {
		values = []string{input.Value}
	}

	// Call the model
	modelResult, err := input.Model.Embed(ctx, model.EmbedCallOptions{
		Values:          values,
		Dimensions:      input.Dimensions,
		ProviderOptions: input.ProviderOptions,
		Headers:         input.Headers,
	})
	if err != nil {
		return nil, err
	}

	// Convert model result to goai result
	embeddings := make([]Embedding, len(modelResult.Embeddings))
	for i, emb := range modelResult.Embeddings {
		embeddings[i] = Embedding{
			Values: emb.Values,
			Index:  emb.Index,
		}
	}

	result := &EmbedResult{
		Embeddings: embeddings,
		Usage: EmbeddingUsage{
			Tokens: modelResult.Usage.Tokens,
		},
		Response: EmbeddingResponseMeta{
			ID:      modelResult.Response.ID,
			Model:   modelResult.Response.Model,
			Headers: modelResult.Response.Headers,
		},
	}

	// Set single embedding if single value was provided
	if input.Value != "" && len(embeddings) > 0 {
		result.Embedding = embeddings[0].Values
	}

	return result, nil
}

// EmbedMany generates embeddings for multiple texts.
// This is an alias for Embed with multiple values.
func EmbedMany(ctx context.Context, input EmbedInput) (*EmbedResult, error) {
	return Embed(ctx, input)
}

// Similarity functions for embeddings.

// CosineSimilarity calculates the cosine similarity between two embeddings.
// Returns a value between -1 and 1, where 1 means identical direction.
func CosineSimilarity(a, b []float64) float64 {
	return model.CosineSimilarity(a, b)
}

// EuclideanDistance calculates the Euclidean distance between two embeddings.
// Returns a non-negative value, where 0 means identical vectors.
func EuclideanDistance(a, b []float64) float64 {
	return model.EuclideanDistance(a, b)
}

// DotProduct calculates the dot product between two embeddings.
func DotProduct(a, b []float64) float64 {
	return model.DotProduct(a, b)
}
