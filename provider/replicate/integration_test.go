package replicate

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
)

func skipIfNoKey(t *testing.T) {
	if os.Getenv("REPLICATE_API_KEY") == "" {
		t.Skip("REPLICATE_API_KEY not set")
	}
}

func getProvider() *Provider {
	return New(Options{APIKey: os.Getenv("REPLICATE_API_KEY")})
}

func TestIntegration_ImageGeneration(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()

	t.Run("black-forest-labs/flux-1.1-pro", func(t *testing.T) {
		m := p.ImageModel("black-forest-labs/flux-1.1-pro")
		ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
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

		t.Logf("Generated %d images", len(result.Images))
	})
}

func TestIntegration_ErrorHandling(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	m := p.ImageModel("invalid/model")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := m.Generate(ctx, model.ImageCallOptions{
		Prompt: "test",
	})

	if err == nil {
		t.Error("expected error with invalid model")
	}
}
