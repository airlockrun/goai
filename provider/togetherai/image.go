package togetherai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
	"net/http"
)

type TogetherImageModel struct {
	id       string
	provider *Provider
}

func (m *TogetherImageModel) ID() string            { return m.id }
func (m *TogetherImageModel) Provider() string      { return "togetherai" }
func (m *TogetherImageModel) MaxImagesPerCall() int { return 1 }
func (m *TogetherImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	if len(opts.Mask) > 0 {
		return nil, errors.New("Together does not support image masks")
	}
	body := map[string]any{"model": m.id, "prompt": opts.Prompt, "response_format": "base64"}
	if opts.N > 1 {
		body["n"] = opts.N
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if opts.Size != "" {
		var width, height int
		if n, err := fmt.Sscanf(opts.Size, "%dx%d", &width, &height); err != nil || n != 2 || width <= 0 || height <= 0 {
			return nil, fmt.Errorf("invalid image size %q", opts.Size)
		}
		body["width"], body["height"] = width, height
	}
	var warnings []stream.Warning
	if opts.AspectRatio != "" {
		warnings = append(warnings, stream.UnsupportedWarning("aspectRatio", "use size"))
	}
	if len(opts.Files) > 0 {
		body["image_url"] = "data:" + http.DetectContentType(opts.Files[0]) + ";base64," + base64.StdEncoding.EncodeToString(opts.Files[0])
	}
	if len(opts.Files) > 1 {
		warnings = append(warnings, stream.UnsupportedWarning("files", "only the first input image is used"))
	}
	for k, v := range opts.ProviderOptions {
		body[k] = v
	}
	var result struct {
		Data []struct {
			Base64 string `json:"b64_json"`
		} `json:"data"`
	}
	headers, err := m.provider.compat.DoJSON(ctx, "/images/generations", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	out := &model.ImageResult{Warnings: warnings, Response: model.ImageResponse{Model: m.id, Headers: headers}}
	for _, image := range result.Data {
		data, err := base64.StdEncoding.DecodeString(image.Base64)
		if err != nil {
			return nil, err
		}
		out.Images = append(out.Images, model.GeneratedImage{Base64: image.Base64, MimeType: http.DetectContentType(data)})
	}
	return out, nil
}
