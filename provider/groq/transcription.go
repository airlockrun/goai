package groq

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

// GroqTranscriptionModel implements the TranscriptionModel interface for Groq.
type GroqTranscriptionModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *GroqTranscriptionModel) ID() string {
	return m.id
}

// Provider returns "groq".
func (m *GroqTranscriptionModel) Provider() string {
	return "groq"
}

// Transcribe transcribes audio to text.
func (m *GroqTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	// Read audio data if provided as reader
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

	// Determine filename
	filename := opts.Filename
	if filename == "" {
		switch opts.MimeType {
		case "audio/wav", "audio/wave":
			filename = "audio.wav"
		case "audio/mp3", "audio/mpeg":
			filename = "audio.mp3"
		case "audio/m4a":
			filename = "audio.m4a"
		case "audio/webm":
			filename = "audio.webm"
		case "audio/ogg":
			filename = "audio.ogg"
		case "audio/flac":
			filename = "audio.flac"
		default:
			filename = "audio.mp3"
		}
	}

	// Build multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
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

	// Add prompt if specified
	if opts.Prompt != "" {
		if err := writer.WriteField("prompt", opts.Prompt); err != nil {
			return nil, fmt.Errorf("failed to write prompt field: %w", err)
		}
	}

	// Request verbose JSON for segments
	if err := writer.WriteField("response_format", "verbose_json"); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.compat.BaseURL()+"/audio/transcriptions", &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.compat.APIKey())
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	// Execute request
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Groq API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var transResp transcriptionResponse
	if err := json.Unmarshal(body, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
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

// Response types

type transcriptionResponse struct {
	Task     string                 `json:"task"`
	Language string                 `json:"language"`
	Duration float64                `json:"duration"`
	Text     string                 `json:"text"`
	Segments []transcriptionSegment `json:"segments,omitempty"`
}

type transcriptionSegment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}
