package model

import (
	"context"

	"github.com/airlockrun/goai/stream"
)

// EmbeddingModel is the interface for text embedding models.
type EmbeddingModel interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// MaxEmbeddingsPerCall returns the maximum number of texts that can be embedded in a single call.
	MaxEmbeddingsPerCall() int

	// Dimensions returns the embedding dimensions (0 if variable or unknown).
	Dimensions() int

	// Embed generates embeddings for the provided texts.
	Embed(ctx context.Context, opts EmbedCallOptions) (*EmbedResult, error)
}

// EmbedCallOptions contains the options for embedding generation.
type EmbedCallOptions struct {
	// Values is the list of texts to embed.
	Values []string

	// Dimensions is the desired embedding dimensions (if model supports it).
	Dimensions *int

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Headers are additional HTTP headers.
	Headers map[string]string
}

// EmbedResult contains the result of an embedding call.
type EmbedResult struct {
	// Embeddings contains the generated embeddings.
	Embeddings []Embedding

	// Usage contains usage information.
	Usage EmbeddingUsage

	// Warnings contains any warnings from the embedding process.
	// Mirrors ai-sdk's EmbeddingModelV3 `warnings: SharedV3Warning[]`.
	Warnings []stream.Warning

	// Response contains provider-specific response data.
	Response EmbeddingResponse
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

// EmbeddingResponse contains provider-specific response metadata.
type EmbeddingResponse struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}

// Similarity functions for embeddings.

// CosineSimilarity calculates the cosine similarity between two embeddings.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// EuclideanDistance calculates the Euclidean distance between two embeddings.
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sqrt(sum)
}

// DotProduct calculates the dot product between two embeddings.
func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// sqrt is a simple square root implementation to avoid importing math.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 100; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
