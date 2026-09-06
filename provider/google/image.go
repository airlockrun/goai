package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type GoogleImageModel struct {
	id       string
	provider *Provider
}

func (m *GoogleImageModel) ID() string            { return m.id }
func (m *GoogleImageModel) Provider() string      { return "google" }
func (m *GoogleImageModel) MaxImagesPerCall() int { return 1 }

func (m *GoogleImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	if !strings.HasPrefix(m.id, "gemini-") {
		return nil, errors.New("Google image models require a gemini- model ID")
	}
	if opts.Mask != nil {
		return nil, errors.New("Gemini image models do not support masks")
	}
	if opts.N > 1 {
		return nil, errors.New("Gemini image models require n <= 1")
	}
	parts := []message.Part{message.TextPart{Text: opts.Prompt}}
	for _, file := range opts.Files {
		parts = append(parts, message.FilePart{MimeType: http.DetectContentType(file), Data: message.FileDataBytes{Data: base64.StdEncoding.EncodeToString(file)}})
	}
	options := make(map[string]any, len(opts.ProviderOptions)+2)
	for k, v := range opts.ProviderOptions {
		options[k] = v
	}
	imageConfig := map[string]any{}
	if config, ok := options["imageConfig"]; ok {
		data, err := json.Marshal(config)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &imageConfig); err != nil {
			return nil, err
		}
		if imageConfig == nil {
			imageConfig = map[string]any{}
		}
	}
	if opts.AspectRatio != "" {
		imageConfig["aspectRatio"] = opts.AspectRatio
	}
	options["imageConfig"] = imageConfig
	options["responseModalities"] = []string{"IMAGE"}
	body, warnings, err := (&GoogleModel{id: m.id, provider: m.provider}).buildRequest(&stream.CallOptions{Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Parts: parts}}}, ProviderOptions: options})
	if err != nil {
		return nil, err
	}
	var request map[string]any
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	if opts.Seed != nil {
		request["generationConfig"].(map[string]any)["seed"] = *opts.Seed
	}
	if search, ok := options["googleSearch"]; ok {
		request["tools"] = []any{map[string]any{"googleSearch": search}}
	}
	if opts.Size != "" {
		warnings = append(warnings, stream.UnsupportedWarning("size", "Use aspectRatio instead."))
	}
	var response geminiStreamChunk
	headers, err := m.provider.post(ctx, m.id, "generateContent", request, opts.Headers, &response)
	if err != nil {
		return nil, err
	}
	if response.PromptFeedback != nil && response.PromptFeedback.BlockReason != "" {
		return nil, goaierrors.ErrContentFiltered
	}
	result := &model.ImageResult{Warnings: warnings, Response: model.ImageResponse{Model: m.id, Headers: headers, Timestamp: time.Now().Unix()}}
	for _, candidate := range response.Candidates {
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				if part.InlineData != nil && strings.HasPrefix(part.InlineData.MimeType, "image/") {
					result.Images = append(result.Images, model.GeneratedImage{Base64: part.InlineData.Data, MimeType: part.InlineData.MimeType})
				}
			}
		}
	}
	if response.UsageMetadata != nil {
		result.Usage = &model.ImageUsage{TotalTokens: response.UsageMetadata.TotalTokenCount}
	}
	metadata := map[string]any{}
	images := make([]map[string]any, len(result.Images))
	for i := range images {
		images[i] = map[string]any{}
	}
	metadata["images"] = images
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]
		if candidate.GroundingMetadata != nil {
			metadata["groundingMetadata"] = mapGroundingMetadata(candidate.GroundingMetadata)
		}
		if candidate.URLContextMetadata != nil {
			metadata["urlContextMetadata"] = mapURLContextMetadata(candidate.URLContextMetadata)
		}
	}
	result.ProviderMetadata = map[string]any{"google": metadata}
	return result, nil
}
