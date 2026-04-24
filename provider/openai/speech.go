package openai

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

// OpenAISpeechModel implements the SpeechModel interface for OpenAI.
type OpenAISpeechModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *OpenAISpeechModel) ID() string {
	return m.id
}

// Provider returns "openai".
func (m *OpenAISpeechModel) Provider() string {
	return "openai"
}

// Generate generates speech from text.
func (m *OpenAISpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	// Build request
	req := speechRequest{
		Model: m.id,
		Input: opts.Text,
		Voice: opts.Voice,
	}

	// Default voice
	if req.Voice == "" {
		req.Voice = "alloy"
	}

	// Mirrors ai-sdk openai-speech-model.ts: only the fixed set of formats
	// is accepted; anything else falls back to mp3 with a warning.
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

	// Speed
	if opts.Speed != nil {
		req.Speed = *opts.Speed
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/audio/speech", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	if m.provider.opts.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", m.provider.opts.Organization)
	}
	if m.provider.opts.Project != "" {
		httpReq.Header.Set("OpenAI-Project", m.provider.opts.Project)
	}
	// Provider-level headers
	for k, v := range m.provider.opts.Headers {
		httpReq.Header.Set(k, v)
	}
	// Request-level headers (override provider headers)
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	// Execute request
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Read audio data
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	// Determine MIME type
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
		Usage: &model.SpeechUsage{
			Characters: len(opts.Text),
		},
		Response: model.SpeechResponse{
			Model: m.id,
		},
	}, nil
}

// Request type

type speechRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format,omitempty"`
	Speed          float64 `json:"speed,omitempty"`
}
