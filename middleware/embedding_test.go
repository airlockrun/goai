package middleware

import (
	"context"
	"testing"

	"github.com/airlockrun/goai/model"
)

// stubEmbeddingModel is a minimal EmbeddingModel used to observe what the
// middleware passes through.
type stubEmbeddingModel struct {
	lastOptions model.EmbedCallOptions
	response    *model.EmbedResult
	err         error
}

func (s *stubEmbeddingModel) ID() string                { return "stub-embed" }
func (s *stubEmbeddingModel) Provider() string          { return "stub" }
func (s *stubEmbeddingModel) MaxEmbeddingsPerCall() int { return 10 }
func (s *stubEmbeddingModel) Dimensions() int           { return 3 }

func (s *stubEmbeddingModel) Embed(_ context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	s.lastOptions = opts
	if s.err != nil {
		return nil, s.err
	}
	if s.response != nil {
		return s.response, nil
	}
	return &model.EmbedResult{}, nil
}

func TestDefaultEmbeddingSettingsMiddleware_MergesDefaultsWithoutOverriding(t *testing.T) {
	stub := &stubEmbeddingModel{}
	mw := &DefaultEmbeddingSettingsMiddleware{
		DefaultHeaders: map[string]string{
			"X-Default":  "yes",
			"X-Override": "from-default",
		},
		DefaultProviderOptions: map[string]any{
			"dimensions": 512,
			"policy":     "default",
		},
	}
	m := WrapEmbeddingModel(stub, mw)

	_, err := m.Embed(context.Background(), model.EmbedCallOptions{
		Values: []string{"hello"},
		Headers: map[string]string{
			"X-Override": "from-caller",
		},
		ProviderOptions: map[string]any{
			"policy": "caller",
		},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	got := stub.lastOptions
	if got.Headers["X-Default"] != "yes" {
		t.Errorf("Headers[X-Default] = %q, want yes", got.Headers["X-Default"])
	}
	if got.Headers["X-Override"] != "from-caller" {
		t.Errorf("Headers[X-Override] = %q, want caller value to win", got.Headers["X-Override"])
	}
	if got.ProviderOptions["dimensions"] != 512 {
		t.Errorf("ProviderOptions[dimensions] = %v, want 512", got.ProviderOptions["dimensions"])
	}
	if got.ProviderOptions["policy"] != "caller" {
		t.Errorf("ProviderOptions[policy] = %v, want caller value to win", got.ProviderOptions["policy"])
	}
}

func TestWrapEmbeddingModel_InvokesWrapEmbed(t *testing.T) {
	stub := &stubEmbeddingModel{response: &model.EmbedResult{Usage: model.EmbeddingUsage{Tokens: 7}}}

	called := 0
	mw := embedFuncMiddleware{
		wrap: func(ctx context.Context, opts *model.EmbedCallOptions, doEmbed EmbedFunc) (*model.EmbedResult, error) {
			called++
			return doEmbed(ctx, opts)
		},
	}
	m := WrapEmbeddingModel(stub, mw)

	result, err := m.Embed(context.Background(), model.EmbedCallOptions{Values: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if called != 1 {
		t.Errorf("wrap called %d times, want 1", called)
	}
	if result.Usage.Tokens != 7 {
		t.Errorf("Usage.Tokens = %d, want 7", result.Usage.Tokens)
	}
}

// embedFuncMiddleware adapts a pure function into the EmbeddingMiddleware
// interface for tests.
type embedFuncMiddleware struct {
	BaseEmbeddingMiddleware
	transform func(*model.EmbedCallOptions) *model.EmbedCallOptions
	wrap      func(context.Context, *model.EmbedCallOptions, EmbedFunc) (*model.EmbedResult, error)
}

func (m embedFuncMiddleware) TransformOptions(opts *model.EmbedCallOptions) *model.EmbedCallOptions {
	if m.transform != nil {
		return m.transform(opts)
	}
	return opts
}

func (m embedFuncMiddleware) WrapEmbed(ctx context.Context, opts *model.EmbedCallOptions, doEmbed EmbedFunc) (*model.EmbedResult, error) {
	if m.wrap != nil {
		return m.wrap(ctx, opts, doEmbed)
	}
	return doEmbed(ctx, opts)
}
