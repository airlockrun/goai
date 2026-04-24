package middleware

import (
	"context"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// fakeProvider implements provider.Provider with stub models for each type.
type fakeProvider struct {
	langCalls  int
	embedCalls int
	imageCalls int
}

func (f *fakeProvider) ID() string                                                { return "fake" }
func (f *fakeProvider) Models() []string                                          { return []string{"x"} }
func (f *fakeProvider) Model(_ string) stream.Model                               { f.langCalls++; return &fakeLangModel{} }
func (f *fakeProvider) LanguageModel(_ string) model.LanguageModel                { f.langCalls++; return &fakeLangModel{} }
func (f *fakeProvider) EmbeddingModel(_ string) model.EmbeddingModel              { f.embedCalls++; return &stubEmbeddingModel{} }
func (f *fakeProvider) ImageModel(_ string) model.ImageModel                      { f.imageCalls++; return &stubImageModel{} }
func (f *fakeProvider) SpeechModel(_ string) model.SpeechModel                    { return nil }
func (f *fakeProvider) TranscriptionModel(_ string) model.TranscriptionModel      { return nil }
func (f *fakeProvider) RerankingModel(_ string) model.RerankingModel              { return nil }

var _ provider.Provider = (*fakeProvider)(nil)

type fakeLangModel struct{}

func (f *fakeLangModel) ID() string       { return "x" }
func (f *fakeLangModel) Provider() string { return "fake" }
func (f *fakeLangModel) Stream(ctx context.Context, opts *stream.CallOptions) (<-chan stream.Event, error) {
	ch := make(chan stream.Event)
	close(ch)
	return ch, nil
}

func TestWrapProvider_WrapsEachModelType(t *testing.T) {
	langSeen := false
	embedSeen := false
	imageSeen := false

	p := WrapProvider(&fakeProvider{}, ProviderMiddlewares{
		Language: []Middleware{&spyLanguageMiddleware{seen: &langSeen}},
		Embedding: []EmbeddingMiddleware{embedFuncMiddleware{
			wrap: func(ctx context.Context, opts *model.EmbedCallOptions, doEmbed EmbedFunc) (*model.EmbedResult, error) {
				embedSeen = true
				return doEmbed(ctx, opts)
			},
		}},
		Image: []ImageMiddleware{imageFuncMiddleware{
			wrap: func(ctx context.Context, opts *model.ImageCallOptions, doGenerate GenerateImageFunc) (*model.ImageResult, error) {
				imageSeen = true
				return doGenerate(ctx, opts)
			},
		}},
	})

	// Language path
	lm := p.LanguageModel("x")
	events, err := lm.Stream(context.Background(), &stream.CallOptions{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range events {
	}
	if !langSeen {
		t.Error("language middleware not invoked")
	}

	// Embedding path
	em := p.EmbeddingModel("x")
	if _, err := em.Embed(context.Background(), model.EmbedCallOptions{Values: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !embedSeen {
		t.Error("embedding middleware not invoked")
	}

	// Image path
	im := p.ImageModel("x")
	if _, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x"}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !imageSeen {
		t.Error("image middleware not invoked")
	}
}

func TestWrapProvider_PassesNilModelsThrough(t *testing.T) {
	// Provider returns nil for Speech/Transcription/Reranking; wrapper must
	// not panic or try to wrap nil.
	p := WrapProvider(&fakeProvider{}, ProviderMiddlewares{
		Language:  []Middleware{&spyLanguageMiddleware{}},
		Embedding: []EmbeddingMiddleware{embedFuncMiddleware{}},
		Image:     []ImageMiddleware{imageFuncMiddleware{}},
	})
	if got := p.SpeechModel("x"); got != nil {
		t.Errorf("SpeechModel = %v, want nil", got)
	}
	if got := p.TranscriptionModel("x"); got != nil {
		t.Errorf("TranscriptionModel = %v, want nil", got)
	}
	if got := p.RerankingModel("x"); got != nil {
		t.Errorf("RerankingModel = %v, want nil", got)
	}
}

type spyLanguageMiddleware struct {
	BaseMiddleware
	seen *bool
}

func (s *spyLanguageMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	if s.seen != nil {
		*s.seen = true
	}
	return doStream(ctx, options)
}
