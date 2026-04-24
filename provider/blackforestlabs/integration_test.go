package blackforestlabs

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("BFL_API_KEY") == "" {
		t.Skip("BFL_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("BFL_API_KEY")})
}

func TestIntegration_ImageGeneration(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("flux-pro-1.1", func(t *testing.T) {
		m := p.ImageModel("flux-pro-1.1")
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		result, err := m.Generate(ctx, model.ImageCallOptions{
			Prompt: "A simple red circle on a white background",
		})

		if err != nil {
			t.Fatalf("Generate image error: %v", err)
		}

		if len(result.Images) == 0 {
			t.Error("expected at least one image")
		}

		if len(result.Images[0].Base64) < 1024 {
			t.Errorf("expected base64 data to be at least 1KB, got %d chars", len(result.Images[0].Base64))
		}

		t.Logf("Generated %d images, first image base64 size: %d chars", len(result.Images), len(result.Images[0].Base64))
	})

	t.Run("flux-dev", func(t *testing.T) {
		m := p.ImageModel("flux-dev")
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		result, err := m.Generate(ctx, model.ImageCallOptions{
			Prompt: "A blue square on a yellow background",
		})

		if err != nil {
			t.Fatalf("Generate image error: %v", err)
		}

		if len(result.Images) == 0 {
			t.Error("expected at least one image")
		}

		t.Logf("Generated image with flux-dev")
	})
}

func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.ImageModel("invalid-model")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := m.Generate(ctx, model.ImageCallOptions{
		Prompt: "test",
	})

	if err == nil {
		t.Error("expected error with invalid model")
	}
}
