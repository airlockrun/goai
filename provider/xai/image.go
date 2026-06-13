package xai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// XaiImageModel implements model.ImageModel for xAI's image generation
// and editing endpoints. Mirrors ai-sdk's XaiImageModel.
// See: ai-sdk/packages/xai/src/xai-image-model.ts
type XaiImageModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *XaiImageModel) ID() string { return m.id }

// Provider returns "xai".
func (m *XaiImageModel) Provider() string { return "xai" }

// MaxImagesPerCall returns the maximum number of images that can be
// generated in a single call. Mirrors ai-sdk (3).
func (m *XaiImageModel) MaxImagesPerCall() int { return 3 }

// Generate generates images based on the provided options.
func (m *XaiImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	var warnings []stream.Warning

	if opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}
	if opts.Seed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("seed", ""))
	}
	if len(opts.Mask) > 0 {
		warnings = append(warnings, stream.UnsupportedWarning("mask", ""))
	}

	xaiOpts, err := provider.ParseProviderOptions[XaiImageOptions](providerOptionsFor(opts.ProviderOptions, "xai"))
	if err != nil {
		return nil, fmt.Errorf("invalid provider options: %w", err)
	}

	hasFiles := len(opts.Files) > 0
	endpoint := "/images/generations"
	if hasFiles {
		endpoint = "/images/edits"
	}

	n := opts.N
	if n <= 0 {
		n = 1
	}

	body := map[string]any{
		"model":           m.id,
		"prompt":          opts.Prompt,
		"n":               n,
		"response_format": "b64_json",
	}

	if opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	}
	if xaiOpts.OutputFormat != "" {
		body["output_format"] = xaiOpts.OutputFormat
	}
	if xaiOpts.SyncMode != nil {
		body["sync_mode"] = *xaiOpts.SyncMode
	}
	if xaiOpts.AspectRatio != "" && opts.AspectRatio == "" {
		body["aspect_ratio"] = xaiOpts.AspectRatio
	}
	if xaiOpts.Resolution != "" {
		body["resolution"] = xaiOpts.Resolution
	}
	if xaiOpts.Quality != "" {
		body["quality"] = xaiOpts.Quality
	}
	if xaiOpts.User != "" {
		body["user"] = xaiOpts.User
	}

	if hasFiles {
		urls := make([]string, len(opts.Files))
		for i, f := range opts.Files {
			urls[i] = bytesToDataURI(f)
		}
		switch len(urls) {
		case 1:
			body["image"] = map[string]any{"url": urls[0], "type": "image_url"}
		default:
			imgs := make([]map[string]any, len(urls))
			for i, u := range urls {
				imgs[i] = map[string]any{"url": u, "type": "image_url"}
			}
			body["images"] = imgs
		}
	}

	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	baseURL := m.provider.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if m.provider.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.provider.apiKey)
	}
	for k, v := range m.provider.headers {
		httpReq.Header.Set(k, v)
	}
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

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
		return nil, fmt.Errorf("xAI API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var parsed xaiImageResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	hasAllBase64 := len(parsed.Data) > 0
	for _, d := range parsed.Data {
		if d.B64JSON == "" {
			hasAllBase64 = false
			break
		}
	}

	images := make([]model.GeneratedImage, len(parsed.Data))
	for i, d := range parsed.Data {
		b64 := d.B64JSON
		if !hasAllBase64 {
			downloaded, derr := m.downloadImage(ctx, d.URL, opts.Headers)
			if derr != nil {
				return nil, derr
			}
			b64 = base64.StdEncoding.EncodeToString(downloaded)
		}
		images[i] = model.GeneratedImage{
			Base64:        b64,
			MimeType:      "image/png",
			RevisedPrompt: d.RevisedPrompt,
		}
	}

	respHeaders := make(map[string]string, len(resp.Header))
	for k, v := range resp.Header {
		if len(v) > 0 {
			respHeaders[k] = v[0]
		}
	}

	xaiMeta := map[string]any{}
	imageMeta := make([]map[string]any, len(parsed.Data))
	for i, d := range parsed.Data {
		item := map[string]any{}
		if d.RevisedPrompt != "" {
			item["revisedPrompt"] = d.RevisedPrompt
		}
		imageMeta[i] = item
	}
	xaiMeta["images"] = imageMeta
	if parsed.Usage != nil && parsed.Usage.CostInUsdTicks != nil {
		xaiMeta["costInUsdTicks"] = *parsed.Usage.CostInUsdTicks
	}

	return &model.ImageResult{
		Images:   images,
		Warnings: warnings,
		Response: model.ImageResponse{
			Model:     m.id,
			Timestamp: time.Now().Unix(),
			Headers:   respHeaders,
		},
		ProviderMetadata: map[string]any{
			"xai": xaiMeta,
		},
	}, nil
}

func (m *XaiImageModel) downloadImage(ctx context.Context, url string, headers map[string]string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build download request: %w", err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read downloaded image: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("image download error (status %d)", resp.StatusCode)
	}
	return body, nil
}

// providerOptionsFor extracts a provider-scoped option map keyed by
// provider ID. Returns nil when the caller supplied no overrides.
func providerOptionsFor(all map[string]any, providerID string) map[string]any {
	if all == nil {
		return nil
	}
	scoped, _ := all[providerID].(map[string]any)
	return scoped
}

// bytesToDataURI converts raw image bytes to a base64-encoded data URI.
// The MIME type is detected from the magic bytes; unknown encodings
// default to image/png. Matches ai-sdk's convertImageModelFileToDataUri
// semantics for Uint8Array inputs.
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
	case len(data) >= 4 && (bytes.HasPrefix(data, []byte("GIF8"))):
		return "image/gif"
	case len(data) >= 2 && bytes.HasPrefix(data, []byte("BM")):
		return "image/bmp"
	default:
		return "image/png"
	}
}

type xaiImageResponse struct {
	Data  []xaiImageData `json:"data"`
	Usage *xaiImageUsage `json:"usage,omitempty"`
}

type xaiImageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

type xaiImageUsage struct {
	CostInUsdTicks *int64 `json:"cost_in_usd_ticks,omitempty"`
}
