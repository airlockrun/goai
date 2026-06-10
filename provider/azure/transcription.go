package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// AzureTranscriptionModel implements the TranscriptionModel interface.
type AzureTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *AzureTranscriptionModel) ID() string       { return m.id }
func (m *AzureTranscriptionModel) Provider() string { return "azure" }

func (m *AzureTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	// Read audio data
	var audioData []byte
	if opts.Audio != nil {
		audioData = opts.Audio
	} else if opts.AudioReader != nil {
		var err error
		audioData, err = io.ReadAll(opts.AudioReader)
		if err != nil {
			return nil, fmt.Errorf("failed to read audio data: %w", err)
		}
	} else {
		return nil, fmt.Errorf("audio data is required")
	}

	// Create multipart form
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	// Add file
	part, err := writer.CreateFormFile("file", "audio.wav")
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(audioData); err != nil {
		return nil, fmt.Errorf("failed to write audio data: %w", err)
	}

	// Add model
	if err := writer.WriteField("model", m.id); err != nil {
		return nil, fmt.Errorf("failed to write model field: %w", err)
	}

	// Add language if specified
	if opts.Language != "" {
		if err := writer.WriteField("language", opts.Language); err != nil {
			return nil, fmt.Errorf("failed to write language field: %w", err)
		}
	}

	// Request verbose JSON for detailed output
	if err := writer.WriteField("response_format", "verbose_json"); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	url := fmt.Sprintf("%s/audio/transcriptions?api-version=%s",
		m.provider.baseURL(m.id), m.provider.opts.APIVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, &body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	if err := m.provider.setAuth(req); err != nil {
		return nil, err
	}
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Azure API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var transResp transcriptionResponse
	if err := json.Unmarshal(respBody, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert segments
	segments := make([]model.TranscriptionSegment, len(transResp.Segments))
	for i, seg := range transResp.Segments {
		segments[i] = model.TranscriptionSegment{
			ID:    seg.ID,
			Text:  seg.Text,
			Start: seg.Start,
			End:   seg.End,
		}
	}

	var duration *float64
	if transResp.Duration > 0 {
		duration = &transResp.Duration
	}

	return &model.TranscriptionResult{
		Text:     transResp.Text,
		Segments: segments,
		Language: transResp.Language,
		Duration: duration,
		Usage: &model.TranscriptionUsage{
			DurationSeconds: transResp.Duration,
		},
		Response: model.TranscriptionResponse{
			Model: m.id,
		},
	}, nil
}

type transcriptionResponse struct {
	Text     string  `json:"text"`
	Language string  `json:"language"`
	Duration float64 `json:"duration"`
	Segments []struct {
		ID    int     `json:"id"`
		Text  string  `json:"text"`
		Start float64 `json:"start"`
		End   float64 `json:"end"`
	} `json:"segments"`
}
