package middleware

import (
	"context"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type stubImageModel struct {
	lastOptions model.ImageCallOptions
	response    *model.ImageResult
}

func (s *stubImageModel) ID() string            { return "stub-image" }
func (s *stubImageModel) Provider() string      { return "stub" }
func (s *stubImageModel) MaxImagesPerCall() int { return 4 }

func (s *stubImageModel) Generate(_ context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	s.lastOptions = opts
	if s.response != nil {
		return s.response, nil
	}
	return &model.ImageResult{}, nil
}

func TestDefaultImageSettingsMiddleware_MergesDefaultsWithoutOverriding(t *testing.T) {
	stub := &stubImageModel{}
	mw := &DefaultImageSettingsMiddleware{
		DefaultHeaders: map[string]string{"X-Default": "yes"},
		DefaultProviderOptions: map[string]any{
			"quality": "standard",
			"size":    "1024x1024",
		},
	}
	m := WrapImageModel(stub, mw)

	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt:          "a cat",
		ProviderOptions: map[string]any{"quality": "hd"},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	got := stub.lastOptions
	if got.Headers["X-Default"] != "yes" {
		t.Errorf("Headers[X-Default] = %q, want yes", got.Headers["X-Default"])
	}
	if got.ProviderOptions["quality"] != "hd" {
		t.Errorf("ProviderOptions[quality] = %v, want caller value to win", got.ProviderOptions["quality"])
	}
	if got.ProviderOptions["size"] != "1024x1024" {
		t.Errorf("ProviderOptions[size] = %v, want default applied", got.ProviderOptions["size"])
	}
}

func TestWrapImageModel_InvokesWrapGenerate(t *testing.T) {
	stub := &stubImageModel{response: &model.ImageResult{Warnings: []stream.Warning{stream.OtherWarning("stub")}}}

	called := 0
	mw := imageFuncMiddleware{
		wrap: func(ctx context.Context, opts *model.ImageCallOptions, doGenerate GenerateImageFunc) (*model.ImageResult, error) {
			called++
			return doGenerate(ctx, opts)
		},
	}
	m := WrapImageModel(stub, mw)

	result, err := m.Generate(context.Background(), model.ImageCallOptions{Prompt: "x"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if called != 1 {
		t.Errorf("wrap called %d times, want 1", called)
	}
	if len(result.Warnings) != 1 || result.Warnings[0].Message != "stub" {
		t.Errorf("Warnings = %v, want [{Other stub}]", result.Warnings)
	}
}

type imageFuncMiddleware struct {
	BaseImageMiddleware
	wrap func(context.Context, *model.ImageCallOptions, GenerateImageFunc) (*model.ImageResult, error)
}

func (m imageFuncMiddleware) WrapGenerate(ctx context.Context, opts *model.ImageCallOptions, doGenerate GenerateImageFunc) (*model.ImageResult, error) {
	if m.wrap != nil {
		return m.wrap(ctx, opts, doGenerate)
	}
	return doGenerate(ctx, opts)
}
