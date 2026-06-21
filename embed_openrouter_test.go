package goai

import (
	"context"
	"os"
	"testing"

	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openai"
)

// TestEmbed_OpenRouterIntegration exercises a real OpenRouter embeddings call
// through the OpenAI-compatible client (base https://openrouter.ai/api/v1 →
// POST /embeddings). OpenRouter has no dedicated goai provider; it's served as
// openai-compat via base URL, so this also proves airlock's runtime path
// (CreateEmbeddingModel falls back to the same openai client).
//
// Opt-in: skipped unless OPENROUTER_API_KEY is set, so normal/offline runs
// don't hit the network or incur cost. Uses a cheap model.
func TestEmbed_OpenRouterIntegration(t *testing.T) {
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("OPENROUTER_API_KEY not set; skipping OpenRouter embedding integration test")
	}

	p := openai.New(provider.Options{
		BaseURL: "https://openrouter.ai/api/v1",
		APIKey:  key,
	})
	m := p.EmbeddingModel("openai/text-embedding-3-small") // cheap embedding model

	res, err := Embed(context.Background(), EmbedInput{
		Model: m,
		Value: "the quick brown fox jumps over the lazy dog",
	})
	if err != nil {
		t.Fatalf("Embed via OpenRouter: %v", err)
	}
	if len(res.Embedding) == 0 {
		t.Fatal("expected a non-empty embedding vector")
	}
	allZero := true
	for _, v := range res.Embedding {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("embedding vector is all zeros")
	}
	t.Logf("OpenRouter embedding OK: %d dimensions", len(res.Embedding))
}
