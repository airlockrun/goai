package fireworks

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultPollIntervalMS = 500
	defaultPollTimeoutMS  = 120_000
)

// backendConfig describes the per-model Fireworks image endpoint shape.
// Mirrors ai-sdk's modelToBackendConfig.
type backendConfig struct {
	urlFormat       string // "workflows" | "workflows_async" | "image_generation"
	supportsSize    bool
	supportsEditing bool
}

var modelToBackendConfig = map[string]backendConfig{
	"accounts/fireworks/models/flux-1-dev-fp8":                   {urlFormat: "workflows"},
	"accounts/fireworks/models/flux-1-schnell-fp8":               {urlFormat: "workflows"},
	"accounts/fireworks/models/flux-kontext-pro":                 {urlFormat: "workflows_async", supportsEditing: true},
	"accounts/fireworks/models/flux-kontext-max":                 {urlFormat: "workflows_async", supportsEditing: true},
	"accounts/fireworks/models/playground-v2-5-1024px-aesthetic": {urlFormat: "image_generation", supportsSize: true},
	"accounts/fireworks/models/japanese-stable-diffusion-xl":     {urlFormat: "image_generation", supportsSize: true},
	"accounts/fireworks/models/playground-v2-1024px-aesthetic":   {urlFormat: "image_generation", supportsSize: true},
	"accounts/fireworks/models/stable-diffusion-xl-1024-v1-0":    {urlFormat: "image_generation", supportsSize: true},
	"accounts/fireworks/models/SSD-1B":                           {urlFormat: "image_generation", supportsSize: true},
}

// FireworksImageModel implements model.ImageModel for Fireworks image
// generation and editing endpoints. Mirrors ai-sdk's FireworksImageModel.
// See: ai-sdk/packages/fireworks/src/fireworks-image-model.ts
type FireworksImageModel struct {
	id               string
	provider         *Provider
	pollIntervalMS   int
	pollTimeoutMS    int
}

// ID returns the model identifier.
func (m *FireworksImageModel) ID() string { return m.id }

// Provider returns "fireworks".
func (m *FireworksImageModel) Provider() string { return "fireworks" }

// MaxImagesPerCall returns the maximum number of images that can be
// generated in a single call. Mirrors ai-sdk (1).
func (m *FireworksImageModel) MaxImagesPerCall() int { return 1 }

// Generate generates images based on the provided options.
func (m *FireworksImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	cfg, known := modelToBackendConfig[m.id]
	if !known {
		cfg = backendConfig{urlFormat: "workflows"}
	}

	var warnings []stream.Warning
	if !cfg.supportsSize && opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}
	if cfg.supportsSize && opts.AspectRatio != "" {
		warnings = append(warnings, stream.UnsupportedWarning("aspectRatio", "This model does not support the `aspectRatio` option."))
	}

	var inputImage string
	if len(opts.Files) > 0 {
		inputImage = bytesToDataURI(opts.Files[0])
		if len(opts.Files) > 1 {
			warnings = append(warnings, stream.OtherWarning("Fireworks only supports a single input image. Additional images are ignored."))
		}
	}
	if len(opts.Mask) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("mask", "Fireworks Kontext models do not support explicit masks. Use the prompt to describe the areas to edit."))
	}

	n := opts.N
	if n <= 0 {
		n = 1
	}

	body := map[string]any{
		"prompt":  opts.Prompt,
		"samples": n,
	}
	if opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if inputImage != "" {
		body["input_image"] = inputImage
	}
	if cfg.supportsSize && opts.Size != "" {
		if w, h, ok := splitSize(opts.Size); ok {
			body["width"] = w
			body["height"] = h
		}
	}
	if extra, ok := opts.ProviderOptions["fireworks"].(map[string]any); ok {
		for k, v := range extra {
			body[k] = v
		}
	}

	baseURL := m.provider.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	if cfg.urlFormat == "workflows_async" {
		return m.doGenerateAsync(ctx, baseURL, body, opts.Headers, warnings)
	}

	return m.doGenerateSync(ctx, baseURL, cfg, body, opts.Headers, warnings)
}

func (m *FireworksImageModel) doGenerateSync(
	ctx context.Context,
	baseURL string,
	cfg backendConfig,
	body map[string]any,
	callHeaders map[string]string,
	warnings []stream.Warning,
) (*model.ImageResult, error) {
	url := getURLForModel(baseURL, m.id, cfg)

	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.applyHeaders(httpReq, callHeaders)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fireworks API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = detectImageMime(respBody)
	}

	respHeaders := headerMap(resp.Header)

	return &model.ImageResult{
		Images: []model.GeneratedImage{{
			Base64:   base64.StdEncoding.EncodeToString(respBody),
			MimeType: mimeType,
		}},
		Warnings: warnings,
		Response: model.ImageResponse{
			Model:     m.id,
			Timestamp: time.Now().Unix(),
			Headers:   respHeaders,
		},
	}, nil
}

func (m *FireworksImageModel) doGenerateAsync(
	ctx context.Context,
	baseURL string,
	body map[string]any,
	callHeaders map[string]string,
	warnings []stream.Warning,
) (*model.ImageResult, error) {
	submitURL := fmt.Sprintf("%s/workflows/%s", baseURL, m.id)
	pollURL := fmt.Sprintf("%s/workflows/%s/get_result", baseURL, m.id)

	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	submitReq, err := http.NewRequestWithContext(ctx, http.MethodPost, submitURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create submit request: %w", err)
	}
	m.applyHeaders(submitReq, callHeaders)

	submitResp, err := http.DefaultClient.Do(submitReq)
	if err != nil {
		return nil, fmt.Errorf("submit request failed: %w", err)
	}
	submitBody, readErr := io.ReadAll(submitResp.Body)
	submitResp.Body.Close()
	if readErr != nil {
		return nil, fmt.Errorf("failed to read submit response: %w", readErr)
	}
	if submitResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fireworks submit error (status %d): %s", submitResp.StatusCode, string(submitBody))
	}

	var submitParsed asyncSubmitResponse
	if err := json.Unmarshal(submitBody, &submitParsed); err != nil {
		return nil, fmt.Errorf("failed to parse submit response: %w", err)
	}
	if submitParsed.RequestID == "" {
		return nil, fmt.Errorf("fireworks submit response missing request_id")
	}

	imageURL, err := m.pollForImageURL(ctx, pollURL, submitParsed.RequestID, callHeaders)
	if err != nil {
		return nil, err
	}

	downloadReq, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create download request: %w", err)
	}
	// Image download does not need Fireworks auth headers, but pass any
	// caller-supplied headers for parity with ai-sdk's getFromApi.
	for k, v := range callHeaders {
		downloadReq.Header.Set(k, v)
	}

	downloadResp, err := http.DefaultClient.Do(downloadReq)
	if err != nil {
		return nil, fmt.Errorf("image download failed: %w", err)
	}
	defer downloadResp.Body.Close()
	imgBytes, err := io.ReadAll(downloadResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read downloaded image: %w", err)
	}
	if downloadResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("image download error (status %d)", downloadResp.StatusCode)
	}

	mimeType := downloadResp.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = detectImageMime(imgBytes)
	}

	return &model.ImageResult{
		Images: []model.GeneratedImage{{
			Base64:   base64.StdEncoding.EncodeToString(imgBytes),
			MimeType: mimeType,
		}},
		Warnings: warnings,
		Response: model.ImageResponse{
			Model:     m.id,
			Timestamp: time.Now().Unix(),
			Headers:   headerMap(downloadResp.Header),
		},
	}, nil
}

func (m *FireworksImageModel) pollForImageURL(
	ctx context.Context,
	pollURL, requestID string,
	callHeaders map[string]string,
) (string, error) {
	intervalMS := m.pollIntervalMS
	if intervalMS <= 0 {
		intervalMS = defaultPollIntervalMS
	}
	timeoutMS := m.pollTimeoutMS
	if timeoutMS <= 0 {
		timeoutMS = defaultPollTimeoutMS
	}
	interval := time.Duration(intervalMS) * time.Millisecond
	maxAttempts := timeoutMS / intervalMS
	if timeoutMS%intervalMS != 0 {
		maxAttempts++
	}
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	pollBody, err := json.Marshal(map[string]string{"id": requestID})
	if err != nil {
		return "", fmt.Errorf("failed to marshal poll body: %w", err)
	}

	for i := 0; i < maxAttempts; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, pollURL, bytes.NewReader(pollBody))
		if err != nil {
			return "", fmt.Errorf("failed to create poll request: %w", err)
		}
		m.applyHeaders(req, callHeaders)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", fmt.Errorf("poll request failed: %w", err)
		}
		raw, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return "", fmt.Errorf("failed to read poll response: %w", readErr)
		}
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("fireworks poll error (status %d): %s", resp.StatusCode, string(raw))
		}

		var parsed asyncPollResponse
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return "", fmt.Errorf("failed to parse poll response: %w", err)
		}

		switch parsed.Status {
		case "Ready":
			if parsed.Result != nil && parsed.Result.Sample != "" {
				return parsed.Result.Sample, nil
			}
			return "", fmt.Errorf("Fireworks poll response is Ready but missing result.sample")
		case "Error", "Failed":
			return "", fmt.Errorf("Fireworks image generation failed with status: %s", parsed.Status)
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(interval):
		}
	}

	return "", fmt.Errorf("Fireworks image generation timed out after %dms", timeoutMS)
}

func (m *FireworksImageModel) applyHeaders(req *http.Request, callHeaders map[string]string) {
	req.Header.Set("Content-Type", "application/json")
	if m.provider.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.provider.apiKey)
	}
	for k, v := range m.provider.headers {
		req.Header.Set(k, v)
	}
	for k, v := range callHeaders {
		req.Header.Set(k, v)
	}
}

// getURLForModel mirrors ai-sdk's getUrlForModel. The caller supplies the
// backend config because doGenerate already looked it up.
func getURLForModel(baseURL, modelID string, cfg backendConfig) string {
	switch cfg.urlFormat {
	case "image_generation":
		return fmt.Sprintf("%s/image_generation/%s", baseURL, modelID)
	case "workflows_async":
		return fmt.Sprintf("%s/workflows/%s", baseURL, modelID)
	default:
		return fmt.Sprintf("%s/workflows/%s/text_to_image", baseURL, modelID)
	}
}

func splitSize(size string) (string, string, bool) {
	parts := strings.SplitN(size, "x", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", false
	}
	return parts[0], parts[1], true
}

func headerMap(h http.Header) map[string]string {
	out := make(map[string]string, len(h))
	for k, v := range h {
		if len(v) > 0 {
			out[k] = v[0]
		}
	}
	return out
}

// bytesToDataURI converts raw image bytes to a base64-encoded data URI.
// The MIME type is detected from the magic bytes; unknown encodings
// default to image/png. Mirrors ai-sdk's convertImageModelFileToDataUri
// for Uint8Array inputs.
func bytesToDataURI(data []byte) string {
	return fmt.Sprintf("data:%s;base64,%s", detectImageMime(data), base64.StdEncoding.EncodeToString(data))
}

func detectImageMime(data []byte) string {
	switch {
	case len(data) >= 4 && bytes.HasPrefix(data, []byte{0x89, 0x50, 0x4E, 0x47}):
		return "image/png"
	case len(data) >= 3 && bytes.HasPrefix(data, []byte{0xFF, 0xD8, 0xFF}):
		return "image/jpeg"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	case len(data) >= 4 && bytes.HasPrefix(data, []byte("GIF8")):
		return "image/gif"
	case len(data) >= 2 && bytes.HasPrefix(data, []byte("BM")):
		return "image/bmp"
	default:
		return "image/png"
	}
}

// asyncSubmitResponse mirrors ai-sdk asyncSubmitResponseSchema.
type asyncSubmitResponse struct {
	RequestID string `json:"request_id"`
}

// asyncPollResponse mirrors ai-sdk asyncPollResponseSchema.
type asyncPollResponse struct {
	ID     string             `json:"id"`
	Status string             `json:"status"`
	Result *asyncPollResult   `json:"result"`
}

type asyncPollResult struct {
	Sample string `json:"sample,omitempty"`
}
