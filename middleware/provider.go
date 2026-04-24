package middleware

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// ProviderMiddlewares bundles per-model-type middleware stacks that get
// applied to every model a wrapped provider hands out. Any field can be nil
// or empty; models without a corresponding middleware are returned unchanged.
//
// Mirrors ai-sdk's wrapProvider({ provider, languageModelMiddleware,
// embeddingModelMiddleware, imageModelMiddleware }).
type ProviderMiddlewares struct {
	Language  []Middleware
	Embedding []EmbeddingMiddleware
	Image     []ImageMiddleware
}

// WrapProvider returns a new Provider that delegates to `p` but wraps every
// model it hands out in the configured middleware stacks. Providers that
// don't support a given model type (return nil) are passed through untouched.
func WrapProvider(p provider.Provider, mws ProviderMiddlewares) provider.Provider {
	return &wrappedProvider{inner: p, mws: mws}
}

type wrappedProvider struct {
	inner provider.Provider
	mws   ProviderMiddlewares
}

func (w *wrappedProvider) ID() string         { return w.inner.ID() }
func (w *wrappedProvider) Models() []string   { return w.inner.Models() }

func (w *wrappedProvider) Model(modelID string) stream.Model {
	m := w.inner.Model(modelID)
	if m == nil || len(w.mws.Language) == 0 {
		return m
	}
	return WrapModel(m, w.mws.Language...)
}

func (w *wrappedProvider) LanguageModel(modelID string) model.LanguageModel {
	m := w.inner.LanguageModel(modelID)
	if m == nil || len(w.mws.Language) == 0 {
		return m
	}
	// model.LanguageModel is a type alias for stream.Model, so WrapModel's
	// return value satisfies both interfaces directly.
	return WrapModel(m, w.mws.Language...)
}

func (w *wrappedProvider) EmbeddingModel(modelID string) model.EmbeddingModel {
	m := w.inner.EmbeddingModel(modelID)
	if m == nil || len(w.mws.Embedding) == 0 {
		return m
	}
	return WrapEmbeddingModel(m, w.mws.Embedding...)
}

func (w *wrappedProvider) ImageModel(modelID string) model.ImageModel {
	m := w.inner.ImageModel(modelID)
	if m == nil || len(w.mws.Image) == 0 {
		return m
	}
	return WrapImageModel(m, w.mws.Image...)
}

// Model types without dedicated middleware still flow through — callers
// wanting to wrap speech/transcription/reranking can write their own wrapper
// until ai-sdk adds those middleware shapes.

func (w *wrappedProvider) SpeechModel(modelID string) model.SpeechModel {
	return w.inner.SpeechModel(modelID)
}

func (w *wrappedProvider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return w.inner.TranscriptionModel(modelID)
}

func (w *wrappedProvider) RerankingModel(modelID string) model.RerankingModel {
	return w.inner.RerankingModel(modelID)
}

