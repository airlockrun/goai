package middleware

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// EmbeddingMiddleware mirrors ai-sdk's EmbeddingModelMiddleware.
// TransformOptions rewrites the call options before they reach the model;
// WrapEmbed is a chance to run code before/after the actual Embed call.
// Either hook may be nil — nil TransformOptions passes options through
// unchanged, nil WrapEmbed passes the call straight to doEmbed.
type EmbeddingMiddleware interface {
	TransformOptions(options *model.EmbedCallOptions) *model.EmbedCallOptions
	WrapEmbed(ctx context.Context, options *model.EmbedCallOptions, doEmbed EmbedFunc) (*model.EmbedResult, error)
}

// EmbedFunc is the inner callable passed to WrapEmbed — invokes the wrapped
// model's Embed method with the (possibly transformed) options.
type EmbedFunc func(ctx context.Context, options *model.EmbedCallOptions) (*model.EmbedResult, error)

// WrapEmbeddingModel returns a new EmbeddingModel that runs the given
// middlewares around each Embed call. First middleware transforms options
// first; last middleware wraps the model directly.
func WrapEmbeddingModel(m model.EmbeddingModel, middlewares ...EmbeddingMiddleware) model.EmbeddingModel {
	if len(middlewares) == 0 {
		return m
	}
	wrapped := m
	for i := len(middlewares) - 1; i >= 0; i-- {
		wrapped = &wrappedEmbeddingModel{inner: wrapped, middleware: middlewares[i]}
	}
	return wrapped
}

type wrappedEmbeddingModel struct {
	inner      model.EmbeddingModel
	middleware EmbeddingMiddleware
}

func (w *wrappedEmbeddingModel) ID() string                { return w.inner.ID() }
func (w *wrappedEmbeddingModel) Provider() string          { return w.inner.Provider() }
func (w *wrappedEmbeddingModel) MaxEmbeddingsPerCall() int { return w.inner.MaxEmbeddingsPerCall() }
func (w *wrappedEmbeddingModel) Dimensions() int           { return w.inner.Dimensions() }

func (w *wrappedEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	transformed := w.middleware.TransformOptions(&opts)
	if transformed == nil {
		transformed = &opts
	}
	doEmbed := func(ctx context.Context, options *model.EmbedCallOptions) (*model.EmbedResult, error) {
		return w.inner.Embed(ctx, *options)
	}
	return w.middleware.WrapEmbed(ctx, transformed, doEmbed)
}

// BaseEmbeddingMiddleware is a pass-through embedding middleware — embed it
// in your own type and override the hook(s) you care about.
type BaseEmbeddingMiddleware struct{}

func (BaseEmbeddingMiddleware) TransformOptions(options *model.EmbedCallOptions) *model.EmbedCallOptions {
	return options
}

func (BaseEmbeddingMiddleware) WrapEmbed(ctx context.Context, options *model.EmbedCallOptions, doEmbed EmbedFunc) (*model.EmbedResult, error) {
	return doEmbed(ctx, options)
}

// DefaultEmbeddingSettingsMiddleware applies default Headers and
// ProviderOptions to every embedding call. Existing values on the incoming
// options take precedence.
type DefaultEmbeddingSettingsMiddleware struct {
	BaseEmbeddingMiddleware

	DefaultHeaders         map[string]string
	DefaultProviderOptions map[string]any
}

func (d *DefaultEmbeddingSettingsMiddleware) TransformOptions(options *model.EmbedCallOptions) *model.EmbedCallOptions {
	result := *options
	if len(d.DefaultHeaders) > 0 {
		if result.Headers == nil {
			result.Headers = make(map[string]string, len(d.DefaultHeaders))
		}
		for k, v := range d.DefaultHeaders {
			if _, ok := result.Headers[k]; !ok {
				result.Headers[k] = v
			}
		}
	}
	if len(d.DefaultProviderOptions) > 0 {
		if result.ProviderOptions == nil {
			result.ProviderOptions = make(map[string]any, len(d.DefaultProviderOptions))
		}
		for k, v := range d.DefaultProviderOptions {
			if _, ok := result.ProviderOptions[k]; !ok {
				result.ProviderOptions[k] = v
			}
		}
	}
	return &result
}
