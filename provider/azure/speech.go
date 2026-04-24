package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// AzureSpeechModel implements the SpeechModel interface.
type AzureSpeechModel struct {
	id       string
	provider *Provider
}

func (m *AzureSpeechModel) ID() string       { return m.id }
func (m *AzureSpeechModel) Provider() string { return "azure" }

func (m *AzureSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	reqBody := map[string]any{
		"model": m.id,
		"input": opts.Text,
	}

	// Voice
	voice := opts.Voice
	if voice == "" {
		voice = "alloy"
	}
	reqBody["voice"] = voice

	// Response format
	responseFormat := opts.OutputFormat
	if responseFormat == "" {
		responseFormat = "mp3"
	}
	reqBody["response_format"] = responseFormat

	// Speed
	if opts.Speed != nil {
		reqBody["speed"] = *opts.Speed
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/audio/speech?api-version=%s",
		m.provider.baseURL(m.id), m.provider.opts.APIVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Azure API error (status %d): %s", resp.StatusCode, string(body))
	}

	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	mimeType := "audio/mpeg"
	switch responseFormat {
	case "opus":
		mimeType = "audio/opus"
	case "aac":
		mimeType = "audio/aac"
	case "flac":
		mimeType = "audio/flac"
	case "wav":
		mimeType = "audio/wav"
	case "pcm":
		mimeType = "audio/pcm"
	}

	return &model.SpeechResult{
		Audio:    audio,
		MimeType: mimeType,
		Usage: &model.SpeechUsage{
			Characters: len(opts.Text),
		},
		Response: model.SpeechResponse{
			Model: m.id,
		},
	}, nil
}
