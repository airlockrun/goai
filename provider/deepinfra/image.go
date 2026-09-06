package deepinfra

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/openaicompat"
	goairesponse "github.com/airlockrun/goai/response"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
)

type DeepInfraImageModel struct {
	id       string
	provider *Provider
}

func (m *DeepInfraImageModel) ID() string            { return m.id }
func (m *DeepInfraImageModel) Provider() string      { return "deepinfra" }
func (m *DeepInfraImageModel) MaxImagesPerCall() int { return 1 }
func (m *DeepInfraImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	var images []string
	var headers map[string]string
	if len(opts.Files) > 0 {
		var body bytes.Buffer
		w := multipart.NewWriter(&body)
		fields := map[string]any{"model": m.id, "prompt": opts.Prompt}
		if opts.N > 0 {
			fields["n"] = opts.N
		}
		if opts.Size != "" {
			fields["size"] = opts.Size
		}
		for k, v := range opts.ProviderOptions {
			fields[k] = v
		}
		for k, v := range fields {
			if err := w.WriteField(k, fmt.Sprint(v)); err != nil {
				return nil, err
			}
		}
		files := append([][]byte(nil), opts.Files...)
		if len(opts.Mask) > 0 {
			files = append(files, opts.Mask)
		}
		for i, data := range files {
			name := "image"
			if i == len(opts.Files) {
				name = "mask"
			}
			h := make(textproto.MIMEHeader)
			h.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{"name": name, "filename": name + ".png"}))
			h.Set("Content-Type", http.DetectContentType(data))
			part, err := w.CreatePart(h)
			if err != nil {
				return nil, err
			}
			if _, err := part.Write(data); err != nil {
				return nil, err
			}
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		resp, err := m.provider.compat.Do(ctx, "/images/edits", w.FormDataContentType(), &body, opts.Headers)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		var result struct {
			Data []struct {
				Base64 string `json:"b64_json"`
			} `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, err
		}
		headers = goairesponse.ExtractResponseHeaders(resp)
		for _, item := range result.Data {
			images = append(images, item.Base64)
		}
	} else {
		if len(opts.Mask) > 0 {
			return nil, errors.New("image mask requires an input image")
		}
		baseURL := strings.TrimSuffix(strings.TrimRight(m.provider.opts.BaseURL, "/"), "/openai")
		if strings.HasSuffix(baseURL, "/v1") {
			baseURL += "/inference"
		}
		client := openaicompat.New(openaicompat.Options{ProviderID: m.Provider(), BaseURL: baseURL, APIKey: m.provider.opts.APIKey, Headers: m.provider.opts.Headers})
		n := opts.N
		if n == 0 {
			n = 1
		}
		body := map[string]any{"prompt": opts.Prompt, "num_images": n}
		if opts.AspectRatio != "" {
			body["aspect_ratio"] = opts.AspectRatio
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
		for k, v := range opts.ProviderOptions {
			body[k] = v
		}
		var result struct {
			Images []string `json:"images"`
		}
		var err error
		headers, err = client.DoJSON(ctx, "/"+m.id, body, &result, opts.Headers)
		if err != nil {
			return nil, err
		}
		images = result.Images
	}
	out := &model.ImageResult{Response: model.ImageResponse{Model: m.id, Headers: headers}}
	for _, image := range images {
		if strings.HasPrefix(image, "data:") {
			_, value, ok := strings.Cut(image, ",")
			if !ok {
				return nil, errors.New("invalid image data URI")
			}
			image = value
		}
		data, err := base64.StdEncoding.DecodeString(image)
		if err != nil {
			return nil, err
		}
		out.Images = append(out.Images, model.GeneratedImage{Base64: image, MimeType: http.DetectContentType(data)})
	}
	return out, nil
}
