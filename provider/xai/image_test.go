package xai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/xai/src/xai-image-model.test.ts.

const xaiImagePrompt = "A cute baby sea otter"

func newImageModel(t *testing.T, baseURL string) *XaiImageModel {
	t.Helper()
	p := New(Options{APIKey: "test-key", BaseURL: baseURL})
	im, ok := p.ImageModel("grok-imagine-image").(*XaiImageModel)
	if !ok {
		t.Fatalf("ImageModel did not return *XaiImageModel")
	}
	return im
}

func TestXaiImage_Constructor(t *testing.T) {
	im := newImageModel(t, "https://api.example.com")
	if im.Provider() != "xai" {
		t.Errorf("Provider(): want xai, got %s", im.Provider())
	}
	if im.ID() != "grok-imagine-image" {
		t.Errorf("ID(): want grok-imagine-image, got %s", im.ID())
	}
	if im.MaxImagesPerCall() != 3 {
		t.Errorf("MaxImagesPerCall(): want 3, got %d", im.MaxImagesPerCall())
	}
}

type capturedCall struct {
	method string
	path   string
	body   map[string]any
	header http.Header
}

func newCapturingServer(t *testing.T, respBody map[string]any, capture *capturedCall) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.method = r.Method
		capture.path = r.URL.Path
		capture.header = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &capture.body)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(respBody)
	}))
}

func TestXaiImage_DoGenerate(t *testing.T) {
	basicResp := map[string]any{"data": []map[string]any{{"b64_json": "dGVzdA=="}}}

	t.Run("sends correct parameters for generation", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      xaiImagePrompt,
			N:           1,
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.method != http.MethodPost {
			t.Errorf("method: want POST, got %s", cap.method)
		}
		if cap.path != "/images/generations" {
			t.Errorf("path: want /images/generations, got %s", cap.path)
		}
		wantBody := map[string]any{
			"model":           "grok-imagine-image",
			"prompt":          xaiImagePrompt,
			"n":               float64(1),
			"response_format": "b64_json",
			"aspect_ratio":    "16:9",
		}
		if !jsonEqual(cap.body, wantBody) {
			t.Errorf("body mismatch:\n got: %v\nwant: %v", cap.body, wantBody)
		}
	})

	t.Run("routes to /images/edits with single file", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		pngBytes := []byte{0x89, 0x50, 0x4E, 0x47}
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "Turn the cat into a dog",
			N:      1,
			Files:  [][]byte{pngBytes},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.path != "/images/edits" {
			t.Errorf("path: want /images/edits, got %s", cap.path)
		}
		img, ok := cap.body["image"].(map[string]any)
		if !ok {
			t.Fatalf("image field missing or wrong type: %v", cap.body["image"])
		}
		wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
		if img["url"] != wantURL {
			t.Errorf("image.url: want %q, got %q", wantURL, img["url"])
		}
		if img["type"] != "image_url" {
			t.Errorf("image.type: want image_url, got %v", img["type"])
		}
		if _, has := cap.body["images"]; has {
			t.Errorf("body should not contain images[] for single file")
		}
	})

	t.Run("sends multiple files as images array", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		png := []byte{0x89, 0x50, 0x4E, 0x47}
		jpg := []byte{0xFF, 0xD8, 0xFF, 0xE0}
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "Combine these images",
			N:      1,
			Files:  [][]byte{png, jpg},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.path != "/images/edits" {
			t.Errorf("path: want /images/edits, got %s", cap.path)
		}
		imgs, ok := cap.body["images"].([]any)
		if !ok || len(imgs) != 2 {
			t.Fatalf("images: want 2-element array, got %v", cap.body["images"])
		}
		first := imgs[0].(map[string]any)
		second := imgs[1].(map[string]any)
		if !strings.HasPrefix(first["url"].(string), "data:image/png;base64,") {
			t.Errorf("images[0].url: want png data URI, got %v", first["url"])
		}
		if !strings.HasPrefix(second["url"].(string), "data:image/jpeg;base64,") {
			t.Errorf("images[1].url: want jpeg data URI, got %v", second["url"])
		}
		if first["type"] != "image_url" || second["type"] != "image_url" {
			t.Errorf("images type mismatch")
		}
		if _, has := cap.body["image"]; has {
			t.Errorf("body should not contain singular image for multi-file edit")
		}
	})

	t.Run("returns base64 images from b64_json response", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result.Images) != 1 {
			t.Fatalf("images: want 1, got %d", len(result.Images))
		}
		if result.Images[0].Base64 != "dGVzdA==" {
			t.Errorf("base64: want dGVzdA==, got %s", result.Images[0].Base64)
		}
	})

	t.Run("downloads url when b64_json missing", func(t *testing.T) {
		imagePayload := []byte("binary-image-bytes")
		imageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(imagePayload)
		}))
		defer imageSrv.Close()

		cap := &capturedCall{}
		srv := newCapturingServer(t, map[string]any{
			"data": []map[string]any{{"url": imageSrv.URL + "/img.png"}},
		}, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		wantB64 := base64.StdEncoding.EncodeToString(imagePayload)
		if result.Images[0].Base64 != wantB64 {
			t.Errorf("downloaded base64: want %q, got %q", wantB64, result.Images[0].Base64)
		}
	})

	t.Run("downloads all when any response entry lacks b64", func(t *testing.T) {
		payload1 := []byte("first-image")
		payload2 := []byte("second-image")
		imageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/img1.png") {
				_, _ = w.Write(payload1)
			} else {
				_, _ = w.Write(payload2)
			}
		}))
		defer imageSrv.Close()

		cap := &capturedCall{}
		srv := newCapturingServer(t, map[string]any{
			"data": []map[string]any{
				{"b64_json": "Zmlyc3Q=", "url": imageSrv.URL + "/img1.png"},
				{"url": imageSrv.URL + "/img2.png"},
			},
		}, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      2,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result.Images) != 2 {
			t.Fatalf("images: want 2, got %d", len(result.Images))
		}
		if got, want := result.Images[0].Base64, base64.StdEncoding.EncodeToString(payload1); got != want {
			t.Errorf("first image base64: want %q, got %q", want, got)
		}
		if got, want := result.Images[1].Base64, base64.StdEncoding.EncodeToString(payload2); got != want {
			t.Errorf("second image base64: want %q, got %q", want, got)
		}
	})

	t.Run("passes headers", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		p := New(Options{
			APIKey:  "test-key",
			BaseURL: srv.URL,
			Headers: map[string]string{"Custom-Provider-Header": "provider-header-value"},
		})
		im := p.ImageModel("grok-imagine-image")
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := cap.header.Get("Custom-Provider-Header"); got != "provider-header-value" {
			t.Errorf("provider header: got %q", got)
		}
		if got := cap.header.Get("Custom-Request-Header"); got != "request-header-value" {
			t.Errorf("request header: got %q", got)
		}
		if got := cap.header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization: got %q", got)
		}
	})

	t.Run("provider options: output_format and sync_mode", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			ProviderOptions: map[string]any{
				"xai": map[string]any{
					"output_format": "jpeg",
					"sync_mode":     true,
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["output_format"] != "jpeg" {
			t.Errorf("output_format: want jpeg, got %v", cap.body["output_format"])
		}
		if cap.body["sync_mode"] != true {
			t.Errorf("sync_mode: want true, got %v", cap.body["sync_mode"])
		}
	})

	t.Run("provider option: resolution", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			ProviderOptions: map[string]any{
				"xai": map[string]any{"resolution": "2k"},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["resolution"] != "2k" {
			t.Errorf("resolution: want 2k, got %v", cap.body["resolution"])
		}
	})

	t.Run("provider option: quality", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			ProviderOptions: map[string]any{
				"xai": map[string]any{"quality": "high"},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["quality"] != "high" {
			t.Errorf("quality: want high, got %v", cap.body["quality"])
		}
	})

	t.Run("provider option: user", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			ProviderOptions: map[string]any{
				"xai": map[string]any{"user": "example-user-123"},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["user"] != "example-user-123" {
			t.Errorf("user: got %v", cap.body["user"])
		}
	})

	t.Run("provider option aspect_ratio fills when top-level missing", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
			ProviderOptions: map[string]any{
				"xai": map[string]any{"aspect_ratio": "4:3"},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["aspect_ratio"] != "4:3" {
			t.Errorf("aspect_ratio: got %v", cap.body["aspect_ratio"])
		}
	})

	t.Run("top-level aspect_ratio wins over provider option", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		_, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      xaiImagePrompt,
			N:           1,
			AspectRatio: "16:9",
			ProviderOptions: map[string]any{
				"xai": map[string]any{"aspect_ratio": "4:3"},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cap.body["aspect_ratio"] != "16:9" {
			t.Errorf("aspect_ratio: want 16:9, got %v", cap.body["aspect_ratio"])
		}
	})

	t.Run("revisedPrompt surfaced on providerMetadata", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, map[string]any{
			"data": []map[string]any{{"b64_json": "dGVzdA==", "revised_prompt": "A revised prompt"}},
		}, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		xai, ok := result.ProviderMetadata["xai"].(map[string]any)
		if !ok {
			t.Fatalf("providerMetadata.xai missing: %v", result.ProviderMetadata)
		}
		imgs, ok := xai["images"].([]map[string]any)
		if !ok {
			t.Fatalf("providerMetadata.xai.images wrong type: %T", xai["images"])
		}
		if len(imgs) != 1 || imgs[0]["revisedPrompt"] != "A revised prompt" {
			t.Errorf("revisedPrompt mismatch: %v", imgs)
		}
		if result.Images[0].RevisedPrompt != "A revised prompt" {
			t.Errorf("GeneratedImage.RevisedPrompt: want %q, got %q", "A revised prompt", result.Images[0].RevisedPrompt)
		}
	})

	t.Run("costInUsdTicks surfaced on providerMetadata", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, map[string]any{
			"data": []map[string]any{{"b64_json": "dGVzdA=="}},
			"usage": map[string]any{
				"cost_in_usd_ticks": 123,
			},
		}, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		xai, ok := result.ProviderMetadata["xai"].(map[string]any)
		if !ok {
			t.Fatalf("providerMetadata.xai missing: %v", result.ProviderMetadata)
		}
		ticks, ok := xai["costInUsdTicks"].(int64)
		if !ok || ticks != 123 {
			t.Errorf("costInUsdTicks: want 123, got %v (type %T)", xai["costInUsdTicks"], xai["costInUsdTicks"])
		}
	})

	t.Run("costInUsdTicks absent when not supplied", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		xai, _ := result.ProviderMetadata["xai"].(map[string]any)
		if _, has := xai["costInUsdTicks"]; has {
			t.Errorf("costInUsdTicks should be absent")
		}
	})
}

func TestXaiImage_Warnings(t *testing.T) {
	basicResp := map[string]any{"data": []map[string]any{{"b64_json": "dGVzdA=="}}}

	tests := []struct {
		name string
		opts model.ImageCallOptions
		want stream.Warning
	}{
		{
			name: "warns on size",
			opts: model.ImageCallOptions{Prompt: xaiImagePrompt, N: 1, Size: "1024x1024"},
			want: stream.Warning{Type: stream.WarningUnsupported, Feature: "size"},
		},
		{
			name: "warns on seed",
			opts: func() model.ImageCallOptions {
				s := int64(42)
				return model.ImageCallOptions{Prompt: xaiImagePrompt, N: 1, Seed: &s}
			}(),
			want: stream.Warning{Type: stream.WarningUnsupported, Feature: "seed"},
		},
		{
			name: "warns on mask",
			opts: model.ImageCallOptions{
				Prompt: "Edit this",
				N:      1,
				Files:  [][]byte{{0x89, 0x50, 0x4E, 0x47}},
				Mask:   []byte{0xFF, 0xFF, 0xFF, 0x00},
			},
			want: stream.Warning{Type: stream.WarningUnsupported, Feature: "mask"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cap := &capturedCall{}
			srv := newCapturingServer(t, basicResp, cap)
			defer srv.Close()

			im := newImageModel(t, srv.URL)
			result, err := im.Generate(context.Background(), tc.opts)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			assertResultWarning(t, result.Warnings, tc.want)
		})
	}

	t.Run("no warnings for clean call", func(t *testing.T) {
		cap := &capturedCall{}
		srv := newCapturingServer(t, basicResp, cap)
		defer srv.Close()

		im := newImageModel(t, srv.URL)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt: xaiImagePrompt,
			N:      1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result.Warnings) != 0 {
			t.Errorf("expected no warnings, got %v", result.Warnings)
		}
	})
}

func TestXaiImage_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid prompt","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()

	im := newImageModel(t, srv.URL)
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: xaiImagePrompt,
		N:      1,
	})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "Invalid prompt") {
		t.Errorf("error should mention Invalid prompt, got %v", err)
	}
}

func TestXaiImage_AbortSignal(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done
	}))
	defer func() {
		close(done)
		srv.Close()
	}()

	im := newImageModel(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := im.Generate(ctx, model.ImageCallOptions{
		Prompt: xaiImagePrompt,
		N:      1,
	})
	if err == nil {
		t.Fatalf("expected abort error, got nil")
	}
}

func TestDetectImageMime(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"png", []byte{0x89, 0x50, 0x4E, 0x47, 0x0D}, "image/png"},
		{"jpeg", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00}, "image/jpeg"},
		{"webp", append([]byte("RIFF\x00\x00\x00\x00WEBP"), 0x00), "image/webp"},
		{"gif", []byte("GIF89a"), "image/gif"},
		{"bmp", []byte("BMxx"), "image/bmp"},
		{"unknown defaults to png", []byte{0x00, 0x01, 0x02, 0x03}, "image/png"},
		{"short buffer", []byte{0x01}, "image/png"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := detectImageMime(tc.data); got != tc.want {
				t.Errorf("detectImageMime(%x) = %q, want %q", tc.data, got, tc.want)
			}
		})
	}
}

func assertResultWarning(t *testing.T, warnings []stream.Warning, want stream.Warning) {
	t.Helper()
	for _, w := range warnings {
		if w.Type == want.Type && w.Feature == want.Feature && w.Message == want.Message {
			return
		}
	}
	t.Errorf("missing warning %+v in %+v", want, warnings)
}

func jsonEqual(a, b map[string]any) bool {
	ab, err := json.Marshal(a)
	if err != nil {
		return false
	}
	bb, err := json.Marshal(b)
	if err != nil {
		return false
	}
	var x, y any
	_ = json.Unmarshal(ab, &x)
	_ = json.Unmarshal(bb, &y)
	return jsonDeepEq(x, y)
}

func jsonDeepEq(a, b any) bool {
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	return string(ab) == string(bb)
}
