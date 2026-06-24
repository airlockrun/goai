package openrouter

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/airlockrun/goai/model"
)

// transcriptionModel implements model.TranscriptionModel against OpenRouter's
// POST {base}/audio/transcriptions. Unlike OpenAI's multipart upload,
// OpenRouter takes a JSON body with base64 audio:
//
//	{"model": "...", "input_audio": {"data": "<base64>", "format": "wav"}}
type transcriptionModel struct {
	id       string
	provider *Provider
}

func (m *transcriptionModel) ID() string       { return m.id }
func (m *transcriptionModel) Provider() string { return "openrouter" }

func (m *transcriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	var audioData []byte
	switch {
	case opts.Audio != nil:
		audioData = opts.Audio
	case opts.AudioReader != nil:
		var err error
		if audioData, err = io.ReadAll(opts.AudioReader); err != nil {
			return nil, fmt.Errorf("failed to read audio data: %w", err)
		}
	default:
		return nil, fmt.Errorf("audio data is required")
	}

	req := transcriptionRequest{
		Model: m.id,
		InputAudio: inputAudio{
			Data:   base64.StdEncoding.EncodeToString(audioData),
			Format: audioFormat(opts.MimeType, opts.Filename),
		},
		Language: opts.Language,
		Prompt:   opts.Prompt,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/audio/transcriptions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.provider.setHeaders(httpReq.Header, opts.Headers)

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
		return nil, fmt.Errorf("OpenRouter API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var tr transcriptionResponse
	if err := json.Unmarshal(respBody, &tr); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	var duration *float64
	if tr.Duration > 0 {
		duration = &tr.Duration
	}
	return &model.TranscriptionResult{
		Text:     tr.Text,
		Language: tr.Language,
		Duration: duration,
		Usage:    &model.TranscriptionUsage{DurationSeconds: tr.Duration},
		Response: model.TranscriptionResponse{Model: m.id},
	}, nil
}

// audioFormat maps a MIME type (or, failing that, a filename extension) to the
// short format token OpenRouter's input_audio.format expects. Defaults to mp3.
func audioFormat(mimeType, filename string) string {
	switch mimeType {
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav"
	case "audio/mp3", "audio/mpeg":
		return "mp3"
	case "audio/mp4", "audio/m4a", "audio/x-m4a":
		return "m4a"
	case "audio/webm":
		return "webm"
	case "audio/ogg":
		return "ogg"
	case "audio/flac":
		return "flac"
	}
	if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(filename)), "."); ext != "" {
		return ext
	}
	return "mp3"
}

type transcriptionRequest struct {
	Model      string     `json:"model"`
	InputAudio inputAudio `json:"input_audio"`
	Language   string     `json:"language,omitempty"`
	Prompt     string     `json:"prompt,omitempty"`
}

type inputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type transcriptionResponse struct {
	Text     string  `json:"text"`
	Language string  `json:"language,omitempty"`
	Duration float64 `json:"duration,omitempty"`
}
