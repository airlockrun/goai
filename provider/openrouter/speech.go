package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// speechModel implements model.SpeechModel against OpenRouter's
// POST {base}/audio/speech, which is OpenAI-compatible (JSON in, audio bytes out).
type speechModel struct {
	id       string
	provider *Provider
}

func (m *speechModel) ID() string       { return m.id }
func (m *speechModel) Provider() string { return "openrouter" }

func (m *speechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	req := speechRequest{Model: m.id, Input: opts.Text, Voice: opts.Voice}
	if req.Voice == "" {
		req.Voice = "alloy"
	}

	var warnings []stream.Warning
	req.ResponseFormat = "mp3"
	if opts.OutputFormat != "" {
		switch opts.OutputFormat {
		case "mp3", "opus", "aac", "flac", "wav", "pcm":
			req.ResponseFormat = opts.OutputFormat
		default:
			warnings = append(warnings, stream.UnsupportedWarning("outputFormat",
				fmt.Sprintf("Unsupported output format: %s. Using mp3 instead.", opts.OutputFormat)))
		}
	}
	if opts.Speed != nil {
		req.Speed = *opts.Speed
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.provider.setHeaders(httpReq.Header, opts.Headers)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenRouter API error (status %d): %s", resp.StatusCode, string(b))
	}

	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	mimeType := "audio/mpeg"
	switch req.ResponseFormat {
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
		Warnings: warnings,
		Usage:    &model.SpeechUsage{Characters: len(opts.Text)},
		Response: model.SpeechResponse{Model: m.id},
	}, nil
}

type speechRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format,omitempty"`
	Speed          float64 `json:"speed,omitempty"`
}
