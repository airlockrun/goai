package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
)

// Translated from ai-sdk/packages/openai/src/image/openai-image-model.test.ts

const testImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

func TestOpenAIImage_DoGenerate(t *testing.T) {
	t.Run("should generate image", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1234567890,
				"data": []map[string]any{
					{"b64_json": testImageBase64, "revised_prompt": "A sunny day at the beach"},
				},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imgModel := provider.ImageModel("dall-e-3")

		result, err := imgModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "sunny day at the beach",
			N:      1,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}

		if result.Images[0].Base64 != testImageBase64 {
			t.Errorf("expected base64 image data")
		}

		if result.Images[0].RevisedPrompt != "A sunny day at the beach" {
			t.Errorf("expected revised prompt, got %s", result.Images[0].RevisedPrompt)
		}
	})

	t.Run("should pass size and quality", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1234567890,
				"data": []map[string]any{
					{"b64_json": testImageBase64},
				},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		imgModel := provider.ImageModel("dall-e-3")

		_, err := imgModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "sunny day at the beach",
			N:      1,
			Size:   "1024x1024",
			ProviderOptions: map[string]any{
				"openai": map[string]any{
					"quality": "hd",
					"style":   "vivid",
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["size"] != "1024x1024" {
			t.Errorf("expected size 1024x1024, got %v", receivedBody["size"])
		}

		if receivedBody["quality"] != "hd" {
			t.Errorf("expected quality hd, got %v", receivedBody["quality"])
		}

		if receivedBody["style"] != "vivid" {
			t.Errorf("expected style vivid, got %v", receivedBody["style"])
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1234567890,
				"data": []map[string]any{
					{"b64_json": testImageBase64},
				},
			})
		}))
		defer server.Close()

		provider := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		imgModel := provider.ImageModel("dall-e-3")

		_, err := imgModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "sunny day at the beach",
			N:      1,
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestOpenAIImage_Warnings(t *testing.T) {
	basicResp := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1234567890,
				"data":    []map[string]any{{"b64_json": testImageBase64}},
			})
		}))
	}

	t.Run("warns on aspectRatio for dall-e-3", func(t *testing.T) {
		srv := basicResp()
		defer srv.Close()
		p := New(provider.Options{APIKey: "k", BaseURL: srv.URL})
		im := p.ImageModel("dall-e-3")
		res, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x", N: 1, AspectRatio: "16:9"})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		testutil.AssertResultWarning(t, res.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "aspectRatio"})
	})

	t.Run("warns on seed", func(t *testing.T) {
		srv := basicResp()
		defer srv.Close()
		p := New(provider.Options{APIKey: "k", BaseURL: srv.URL})
		im := p.ImageModel("dall-e-3")
		seed := int64(42)
		res, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x", N: 1, Seed: &seed})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		testutil.AssertResultWarning(t, res.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "seed"})
	})

	t.Run("no warning when aspectRatio is applied on gpt-image-1", func(t *testing.T) {
		srv := basicResp()
		defer srv.Close()
		p := New(provider.Options{APIKey: "k", BaseURL: srv.URL})
		im := p.ImageModel("gpt-image-1")
		res, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x", N: 1, AspectRatio: "1024x1024"})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		for _, w := range res.Warnings {
			if w.Feature == "aspectRatio" {
				t.Errorf("did not expect aspectRatio warning on gpt-image-1, got %+v", res.Warnings)
			}
		}
	})

	// Mirrors ai-sdk's OpenAIImageModelId family + hasDefaultResponseFormat
	// (PR #14680, packages/openai/src/image/openai-image-options.ts).
	t.Run("gpt-image family is recognized for capabilities and wire shape", func(t *testing.T) {
		family := []string{"gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2", "chatgpt-image-latest"}

		for _, id := range family {
			id := id
			t.Run(id, func(t *testing.T) {
				var captured map[string]any
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					json.NewDecoder(r.Body).Decode(&captured)
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(map[string]any{
						"created": 1,
						"data":    []map[string]any{{"b64_json": testImageBase64}},
					})
				}))
				defer server.Close()

				p := New(provider.Options{APIKey: "k", BaseURL: server.URL})
				im := p.ImageModel(id)

				if got := im.(*OpenAIImageModel).MaxImagesPerCall(); got != 10 {
					t.Errorf("%s MaxImagesPerCall = %d, want 10", id, got)
				}

				res, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x", N: 1, AspectRatio: "1024x1024"})
				if err != nil {
					t.Fatalf("Generate: %v", err)
				}
				for _, w := range res.Warnings {
					if w.Feature == "aspectRatio" {
						t.Errorf("did not expect aspectRatio warning on %s, got %+v", id, res.Warnings)
					}
				}
				if _, has := captured["response_format"]; has {
					t.Errorf("%s wire body should omit response_format, got %v", id, captured["response_format"])
				}
				if captured["size"] != "1024x1024" {
					t.Errorf("%s wire body size = %v, want 1024x1024", id, captured["size"])
				}
			})
		}
	})

	t.Run("dall-e-3 still sends response_format=b64_json", func(t *testing.T) {
		var captured map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1,
				"data":    []map[string]any{{"b64_json": testImageBase64}},
			})
		}))
		defer server.Close()

		p := New(provider.Options{APIKey: "k", BaseURL: server.URL})
		im := p.ImageModel("dall-e-3")
		_, err := im.Generate(context.Background(), model.ImageCallOptions{Prompt: "x", N: 1})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if captured["response_format"] != "b64_json" {
			t.Errorf("dall-e-3 wire body response_format = %v, want b64_json", captured["response_format"])
		}
	})
}
