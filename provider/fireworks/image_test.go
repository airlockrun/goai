package fireworks

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/fireworks/src/fireworks-image-model.test.ts.

const imagePrompt = "A cute baby sea otter"

func newFireworksImageModel(t *testing.T, modelID, baseURL string, opts Options) *FireworksImageModel {
	t.Helper()
	opts.BaseURL = baseURL
	if opts.APIKey == "" {
		opts.APIKey = "test-key"
	}
	p := New(opts)
	im, ok := p.ImageModel(modelID).(*FireworksImageModel)
	if !ok {
		t.Fatalf("ImageModel(%q) did not return *FireworksImageModel", modelID)
	}
	return im
}

func TestFireworksImage_Constructor(t *testing.T) {
	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-1-dev-fp8",
		"https://api.example.com",
		Options{},
	)
	if im.Provider() != "fireworks" {
		t.Errorf("Provider(): want fireworks, got %s", im.Provider())
	}
	if im.ID() != "accounts/fireworks/models/flux-1-dev-fp8" {
		t.Errorf("ID(): got %s", im.ID())
	}
	if im.MaxImagesPerCall() != 1 {
		t.Errorf("MaxImagesPerCall(): want 1, got %d", im.MaxImagesPerCall())
	}
}

type capturedRequest struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

// syncImageServer returns a test server that responds with binary image bytes.
func syncImageServer(t *testing.T, imgBody []byte, captures *[]capturedRequest) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c := capturedRequest{
			method: r.Method,
			path:   r.URL.Path,
			header: r.Header.Clone(),
		}
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &c.body)
		}
		mu.Lock()
		*captures = append(*captures, c)
		mu.Unlock()
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(imgBody)
	}))
}

func TestFireworksImage_SyncWorkflowsURL(t *testing.T) {
	var captures []capturedRequest
	srv := syncImageServer(t, []byte("test-binary-content"), &captures)
	defer srv.Close()

	im := newFireworksImageModel(t, "accounts/fireworks/models/flux-1-dev-fp8", srv.URL, Options{})
	seed := int64(42)
	result, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt:      imagePrompt,
		N:           1,
		AspectRatio: "16:9",
		Seed:        &seed,
		ProviderOptions: map[string]any{
			"fireworks": map[string]any{"additional_param": "value"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(captures) != 1 {
		t.Fatalf("expected 1 request, got %d", len(captures))
	}
	cap := captures[0]
	if cap.method != http.MethodPost {
		t.Errorf("method: want POST, got %s", cap.method)
	}
	wantPath := "/workflows/accounts/fireworks/models/flux-1-dev-fp8/text_to_image"
	if cap.path != wantPath {
		t.Errorf("path: want %s, got %s", wantPath, cap.path)
	}
	if cap.body["prompt"] != imagePrompt {
		t.Errorf("prompt: got %v", cap.body["prompt"])
	}
	if cap.body["samples"] != float64(1) {
		t.Errorf("samples: want 1, got %v", cap.body["samples"])
	}
	if cap.body["aspect_ratio"] != "16:9" {
		t.Errorf("aspect_ratio: got %v", cap.body["aspect_ratio"])
	}
	if cap.body["seed"] != float64(42) {
		t.Errorf("seed: want 42, got %v", cap.body["seed"])
	}
	if cap.body["additional_param"] != "value" {
		t.Errorf("additional_param merged from providerOptions: got %v", cap.body["additional_param"])
	}
	if len(result.Images) != 1 {
		t.Fatalf("images: want 1, got %d", len(result.Images))
	}
	wantB64 := base64.StdEncoding.EncodeToString([]byte("test-binary-content"))
	if result.Images[0].Base64 != wantB64 {
		t.Errorf("base64: want %q, got %q", wantB64, result.Images[0].Base64)
	}
}

func TestFireworksImage_ImageGenerationURLAndSize(t *testing.T) {
	var captures []capturedRequest
	srv := syncImageServer(t, []byte("test-binary-content"), &captures)
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/playground-v2-5-1024px-aesthetic",
		srv.URL,
		Options{},
	)
	seed := int64(42)
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
		Size:   "1024x768",
		Seed:   &seed,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cap := captures[0]
	wantPath := "/image_generation/accounts/fireworks/models/playground-v2-5-1024px-aesthetic"
	if cap.path != wantPath {
		t.Errorf("path: want %s, got %s", wantPath, cap.path)
	}
	if cap.body["width"] != "1024" {
		t.Errorf("width: want 1024, got %v", cap.body["width"])
	}
	if cap.body["height"] != "768" {
		t.Errorf("height: want 768, got %v", cap.body["height"])
	}
	if cap.body["samples"] != float64(1) {
		t.Errorf("samples: want 1, got %v", cap.body["samples"])
	}
}

func TestFireworksImage_Headers(t *testing.T) {
	var captures []capturedRequest
	srv := syncImageServer(t, []byte("ok"), &captures)
	defer srv.Close()

	p := New(Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Headers: map[string]string{"Custom-Provider-Header": "provider-header-value"},
	})
	im := p.ImageModel("accounts/fireworks/models/flux-1-dev-fp8")

	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
		Headers: map[string]string{
			"Custom-Request-Header": "request-header-value",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	h := captures[0].header
	if got := h.Get("Custom-Provider-Header"); got != "provider-header-value" {
		t.Errorf("provider header: got %q", got)
	}
	if got := h.Get("Custom-Request-Header"); got != "request-header-value" {
		t.Errorf("request header: got %q", got)
	}
	if got := h.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("authorization: got %q", got)
	}
	if got := h.Get("Content-Type"); got != "application/json" {
		t.Errorf("content-type: got %q", got)
	}
}

func TestFireworksImage_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("Bad Request"))
	}))
	defer srv.Close()

	im := newFireworksImageModel(t, "accounts/fireworks/models/flux-1-dev-fp8", srv.URL, Options{})
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
	})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "Bad Request") {
		t.Errorf("error should mention Bad Request, got %v", err)
	}
}

func TestFireworksImage_Warnings(t *testing.T) {
	imgBody := []byte("ok")

	t.Run("size warning on workflow model", func(t *testing.T) {
		var captures []capturedRequest
		srv := syncImageServer(t, imgBody, &captures)
		defer srv.Close()

		im := newFireworksImageModel(t, "accounts/fireworks/models/flux-1-dev-fp8", srv.URL, Options{})
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      imagePrompt,
			N:           1,
			Size:        "1024x1024",
			AspectRatio: "1:1",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertResultWarning(t, result.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "size"})
	})

	t.Run("aspectRatio warning on size-supporting model", func(t *testing.T) {
		var captures []capturedRequest
		srv := syncImageServer(t, imgBody, &captures)
		defer srv.Close()

		im := newFireworksImageModel(t,
			"accounts/fireworks/models/playground-v2-5-1024px-aesthetic",
			srv.URL,
			Options{},
		)
		result, err := im.Generate(context.Background(), model.ImageCallOptions{
			Prompt:      imagePrompt,
			N:           1,
			Size:        "1024x1024",
			AspectRatio: "1:1",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertResultWarning(t, result.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "aspectRatio"})
	})
}

// asyncRoundTripper simulates Fireworks async workflow endpoints.
type asyncRoundTripper struct {
	submitURL string
	pollURL   string
	imageURL  string

	submitCaptures []capturedRequest
	pollCaptures   []capturedRequest
	getCaptures    []capturedRequest

	pollResponses [][]byte
	pollIndex     int32

	imageBytes []byte

	mu sync.Mutex
}

func (rt *asyncRoundTripper) capture(r *http.Request) capturedRequest {
	raw, _ := io.ReadAll(r.Body)
	c := capturedRequest{method: r.Method, path: r.URL.Path, header: r.Header.Clone()}
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &c.body)
	}
	return c
}

// startAsyncTestServer boots an httptest server whose URL is set by
// httptest itself; the returned rt wires submit/poll/image URLs against
// that base.
func startAsyncTestServer(t *testing.T, modelID string, imageBytes []byte, pollResponses [][]byte) (*httptest.Server, *asyncRoundTripper) {
	t.Helper()
	rt := &asyncRoundTripper{
		imageBytes:    imageBytes,
		pollResponses: pollResponses,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fullURL := "http://" + r.Host + r.URL.RequestURI()
		cap := rt.capture(r)
		rt.mu.Lock()
		switch fullURL {
		case rt.submitURL:
			rt.submitCaptures = append(rt.submitCaptures, cap)
			rt.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"request_id":"test-request-123"}`))
		case rt.pollURL:
			rt.pollCaptures = append(rt.pollCaptures, cap)
			idx := int(atomic.AddInt32(&rt.pollIndex, 1)) - 1
			var body []byte
			if idx < len(rt.pollResponses) {
				body = rt.pollResponses[idx]
			} else if len(rt.pollResponses) > 0 {
				body = rt.pollResponses[len(rt.pollResponses)-1]
			} else {
				body = []byte("{}")
			}
			rt.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(body)
		case rt.imageURL:
			rt.getCaptures = append(rt.getCaptures, cap)
			rt.mu.Unlock()
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(rt.imageBytes)
		default:
			rt.mu.Unlock()
			http.Error(w, "not found: "+fullURL, http.StatusNotFound)
		}
	}))
	rt.submitURL = srv.URL + "/workflows/" + modelID
	rt.pollURL = srv.URL + "/workflows/" + modelID + "/get_result"
	rt.imageURL = srv.URL + "/result/image.png"
	return srv, rt
}

func readyPollResponse(sampleURL string) []byte {
	return []byte(`{"id":"test-request-123","status":"Ready","result":{"sample":"` + sampleURL + `"}}`)
}

func pendingPollResponse() []byte {
	return []byte(`{"id":"test-request-123","status":"Pending","result":null}`)
}

func TestFireworksImage_AsyncSubmitAndPoll(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("async-image-content"),
		nil,
	)
	rt.pollResponses = [][]byte{readyPollResponse(rt.imageURL)}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	seed := int64(42)
	result, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt:      imagePrompt,
		N:           1,
		AspectRatio: "16:9",
		Seed:        &seed,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := len(rt.submitCaptures); got != 1 {
		t.Fatalf("submit calls: want 1, got %d", got)
	}
	submitBody := rt.submitCaptures[0].body
	if submitBody["prompt"] != imagePrompt {
		t.Errorf("submit prompt: got %v", submitBody["prompt"])
	}
	if submitBody["aspect_ratio"] != "16:9" {
		t.Errorf("submit aspect_ratio: got %v", submitBody["aspect_ratio"])
	}
	if submitBody["seed"] != float64(42) {
		t.Errorf("submit seed: got %v", submitBody["seed"])
	}
	if submitBody["samples"] != float64(1) {
		t.Errorf("submit samples: got %v", submitBody["samples"])
	}

	if got := len(rt.pollCaptures); got != 1 {
		t.Fatalf("poll calls: want 1, got %d", got)
	}
	if rt.pollCaptures[0].body["id"] != "test-request-123" {
		t.Errorf("poll id: got %v", rt.pollCaptures[0].body["id"])
	}

	if got := len(rt.getCaptures); got != 1 {
		t.Fatalf("image download calls: want 1, got %d", got)
	}

	if len(result.Images) != 1 {
		t.Fatalf("images: want 1, got %d", len(result.Images))
	}
	wantB64 := base64.StdEncoding.EncodeToString([]byte("async-image-content"))
	if result.Images[0].Base64 != wantB64 {
		t.Errorf("base64: want %q, got %q", wantB64, result.Images[0].Base64)
	}
}

func TestFireworksImage_AsyncPollUntilReady(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("async-image-content"),
		nil,
	)
	rt.pollResponses = [][]byte{
		pendingPollResponse(),
		pendingPollResponse(),
		readyPollResponse(rt.imageURL),
	}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	result, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(rt.pollCaptures); got != 3 {
		t.Errorf("poll count: want 3, got %d", got)
	}
	if len(result.Images) != 1 {
		t.Fatalf("images: want 1, got %d", len(result.Images))
	}
}

func TestFireworksImage_AsyncErrors(t *testing.T) {
	errorResp := []byte(`{"id":"test-request-123","status":"Error","result":null}`)
	failedResp := []byte(`{"id":"test-request-123","status":"Failed","result":null}`)
	readyNoSampleResp := []byte(`{"id":"test-request-123","status":"Ready","result":{}}`)

	tests := []struct {
		name     string
		resp     []byte
		errorSub string
	}{
		{"status Error", errorResp, "Fireworks image generation failed with status: Error"},
		{"status Failed", failedResp, "Fireworks image generation failed with status: Failed"},
		{"Ready without sample", readyNoSampleResp, "Fireworks poll response is Ready but missing result.sample"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, rt := startAsyncTestServer(t,
				"accounts/fireworks/models/flux-kontext-pro",
				[]byte("ignored"),
				nil,
			)
			rt.pollResponses = [][]byte{tc.resp}
			defer srv.Close()

			im := newFireworksImageModel(t,
				"accounts/fireworks/models/flux-kontext-pro",
				srv.URL,
				Options{PollIntervalMS: 1, PollTimeoutMS: 500},
			)
			_, err := im.Generate(context.Background(), model.ImageCallOptions{
				Prompt: imagePrompt,
				N:      1,
			})
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.errorSub) {
				t.Errorf("error should contain %q, got %v", tc.errorSub, err)
			}
		})
	}
}

func TestFireworksImage_AsyncTimeout(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("ignored"),
		nil,
	)
	rt.pollResponses = [][]byte{pendingPollResponse()}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 5},
	)
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
	})
	if err == nil {
		t.Fatalf("expected timeout error, got nil")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("expected timeout error, got %v", err)
	}
}

func TestFireworksImage_AsyncInputImageAndMask(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("edited-image"),
		nil,
	)
	rt.pollResponses = [][]byte{readyPollResponse(rt.imageURL)}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	pngData := []byte{0x89, 0x50, 0x4E, 0x47}
	maskData := []byte{0xFF, 0xFF, 0xFF, 0x00}

	result, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "Turn cat into dog",
		N:      1,
		Files:  [][]byte{pngData},
		Mask:   maskData,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	wantInput := bytesToDataURI(pngData)
	if rt.submitCaptures[0].body["input_image"] != wantInput {
		t.Errorf("input_image: want %q, got %q", wantInput, rt.submitCaptures[0].body["input_image"])
	}
	assertResultWarning(t, result.Warnings, stream.Warning{Type: stream.WarningUnsupported, Feature: "mask"})
}

func TestFireworksImage_MultipleFilesWarning(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("edited"),
		nil,
	)
	rt.pollResponses = [][]byte{readyPollResponse(rt.imageURL)}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	png := []byte{0x89, 0x50, 0x4E, 0x47}

	result, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "Edit images",
		N:      1,
		Files:  [][]byte{png, png},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertResultWarning(t, result.Warnings, stream.Warning{
		Type:    stream.WarningOther,
		Message: "Fireworks only supports a single input image. Additional images are ignored.",
	})
}

func TestFireworksImage_AsyncProviderOptionsMerged(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("edited"),
		nil,
	)
	rt.pollResponses = [][]byte{readyPollResponse(rt.imageURL)}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: imagePrompt,
		N:      1,
		ProviderOptions: map[string]any{
			"fireworks": map[string]any{
				"safety_tolerance": float64(6),
				"input_image":      "base64-image-data",
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := rt.submitCaptures[0].body
	if body["safety_tolerance"] != float64(6) {
		t.Errorf("safety_tolerance: got %v", body["safety_tolerance"])
	}
	if body["input_image"] != "base64-image-data" {
		t.Errorf("input_image override from providerOptions: got %v", body["input_image"])
	}
}

func TestFireworksImage_AsyncURLShape(t *testing.T) {
	srv, rt := startAsyncTestServer(t,
		"accounts/fireworks/models/flux-kontext-pro",
		[]byte("edited"),
		nil,
	)
	rt.pollResponses = [][]byte{readyPollResponse(rt.imageURL)}
	defer srv.Close()

	im := newFireworksImageModel(t,
		"accounts/fireworks/models/flux-kontext-pro",
		srv.URL,
		Options{PollIntervalMS: 1, PollTimeoutMS: 500},
	)
	png := []byte{0x89, 0x50, 0x4E, 0x47}
	_, err := im.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "Edit this image",
		N:      1,
		Files:  [][]byte{png},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	wantPath := "/workflows/accounts/fireworks/models/flux-kontext-pro"
	if rt.submitCaptures[0].path != wantPath {
		t.Errorf("submit URL: want %s, got %s (should not include text_to_image)", wantPath, rt.submitCaptures[0].path)
	}
}

func TestFireworksImage_DetectMime(t *testing.T) {
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
