package google

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
)

// Translated from ai-sdk/packages/google/src/google-generative-ai-image-model.test.ts.
// Only the warnings paths are covered here; the request-body shape is validated
// indirectly through the integration tests.
func TestGoogleImage_Warnings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{
				"bytesBase64Encoded": "dGVzdA==",
				"mimeType":           "image/png",
			}},
		})
	}))
	defer srv.Close()

	p := New(Options{APIKey: "k", BaseURL: srv.URL})
	im := p.ImageModel("imagen-3.0-generate-001")

	t.Run("warns on size", func(t *testing.T) {
		res, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "x",
			N:      1,
			Size:   "1024x1024",
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		testutil.AssertResultWarning(t, res.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "size"})
	})

	t.Run("warns on seed", func(t *testing.T) {
		seed := int64(42)
		res, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "x",
			N:      1,
			Seed:   &seed,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		testutil.AssertResultWarning(t, res.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "seed"})
	})

	t.Run("no warnings for clean call", func(t *testing.T) {
		res, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      "x",
			N:           1,
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(res.Warnings) != 0 {
			t.Errorf("expected no warnings, got %v", res.Warnings)
		}
	})
}
