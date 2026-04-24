package middleware

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// ImageMiddleware mirrors ai-sdk's ImageModelMiddleware. Hooks run around
// Generate calls; nil implementations fall through.
type ImageMiddleware interface {
	TransformOptions(options *model.ImageCallOptions) *model.ImageCallOptions
	WrapGenerate(ctx context.Context, options *model.ImageCallOptions, doGenerate GenerateImageFunc) (*model.ImageResult, error)
}

// GenerateImageFunc is the inner callable passed to WrapGenerate.
type GenerateImageFunc func(ctx context.Context, options *model.ImageCallOptions) (*model.ImageResult, error)

// WrapImageModel returns a new ImageModel with the given middlewares applied.
// First middleware transforms options first; last wraps the model directly.
func WrapImageModel(m model.ImageModel, middlewares ...ImageMiddleware) model.ImageModel {
	if len(middlewares) == 0 {
		return m
	}
	wrapped := m
	for i := len(middlewares) - 1; i >= 0; i-- {
		wrapped = &wrappedImageModel{inner: wrapped, middleware: middlewares[i]}
	}
	return wrapped
}

type wrappedImageModel struct {
	inner      model.ImageModel
	middleware ImageMiddleware
}

func (w *wrappedImageModel) ID() string            { return w.inner.ID() }
func (w *wrappedImageModel) Provider() string      { return w.inner.Provider() }
func (w *wrappedImageModel) MaxImagesPerCall() int { return w.inner.MaxImagesPerCall() }

func (w *wrappedImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	transformed := w.middleware.TransformOptions(&opts)
	if transformed == nil {
		transformed = &opts
	}
	doGenerate := func(ctx context.Context, options *model.ImageCallOptions) (*model.ImageResult, error) {
		return w.inner.Generate(ctx, *options)
	}
	return w.middleware.WrapGenerate(ctx, transformed, doGenerate)
}

// BaseImageMiddleware is a pass-through image middleware.
type BaseImageMiddleware struct{}

func (BaseImageMiddleware) TransformOptions(options *model.ImageCallOptions) *model.ImageCallOptions {
	return options
}

func (BaseImageMiddleware) WrapGenerate(ctx context.Context, options *model.ImageCallOptions, doGenerate GenerateImageFunc) (*model.ImageResult, error) {
	return doGenerate(ctx, options)
}

// DefaultImageSettingsMiddleware applies default Headers and ProviderOptions
// to every image generation call. Existing values on the incoming options
// take precedence.
type DefaultImageSettingsMiddleware struct {
	BaseImageMiddleware

	DefaultHeaders         map[string]string
	DefaultProviderOptions map[string]any
}

func (d *DefaultImageSettingsMiddleware) TransformOptions(options *model.ImageCallOptions) *model.ImageCallOptions {
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
