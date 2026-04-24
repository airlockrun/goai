// Package revai provides a Rev.ai provider implementation for transcription.
package revai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.rev.ai/speechtotext/v1"
)

// Options contains configuration for the Rev.ai provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Rev.ai provider.
type Provider struct {
	opts Options
}

// New creates a new Rev.ai provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "revai" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel       { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &RevAITranscriptionModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) Models() []string {
	return []string{
		"default",
		"machine",
	}
}

var _ provider.Provider = (*Provider)(nil)

// RevAITranscriptionModel implements the TranscriptionModel interface.
type RevAITranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *RevAITranscriptionModel) ID() string       { return m.id }
func (m *RevAITranscriptionModel) Provider() string { return "revai" }

func (m *RevAITranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
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
	} else if opts.AudioURL != "" {
		return m.transcribeFromURL(ctx, opts.AudioURL, opts)
	} else {
		return nil, fmt.Errorf("audio data is required")
	}

	return m.transcribeFromData(ctx, audioData, opts)
}

func (m *RevAITranscriptionModel) transcribeFromData(ctx context.Context, audioData []byte, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	// Create multipart form
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	// Add media file
	part, err := writer.CreateFormFile("media", opts.Filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(audioData); err != nil {
		return nil, fmt.Errorf("failed to write audio data: %w", err)
	}

	// Add options as JSON
	jobOpts := map[string]any{}
	if opts.Language != "" {
		jobOpts["language"] = opts.Language
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		jobOpts[k] = v
	}

	if len(jobOpts) > 0 {
		optsBytes, _ := json.Marshal(jobOpts)
		if err := writer.WriteField("options", string(optsBytes)); err != nil {
			return nil, fmt.Errorf("failed to write options: %w", err)
		}
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	url := fmt.Sprintf("%s/jobs", m.provider.opts.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, &body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
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

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("Rev.ai API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var jobResp struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(respBody, &jobResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return m.pollForResult(ctx, jobResp.ID)
}

func (m *RevAITranscriptionModel) transcribeFromURL(ctx context.Context, audioURL string, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	reqBody := map[string]any{
		"media_url": audioURL,
	}

	if opts.Language != "" {
		reqBody["language"] = opts.Language
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/jobs", m.provider.opts.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("Rev.ai API error (status %d): %s", resp.StatusCode, string(body))
	}

	var jobResp struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &jobResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return m.pollForResult(ctx, jobResp.ID)
}

func (m *RevAITranscriptionModel) pollForResult(ctx context.Context, jobID string) (*model.TranscriptionResult, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		// Check job status
		statusURL := fmt.Sprintf("%s/jobs/%s", m.provider.opts.BaseURL, jobID)

		req, err := http.NewRequestWithContext(ctx, "GET", statusURL, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request failed: %w", err)
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}

		var statusResp struct {
			ID              string  `json:"id"`
			Status          string  `json:"status"`
			DurationSeconds float64 `json:"duration_seconds"`
			Language        string  `json:"language"`
		}
		if err := json.Unmarshal(body, &statusResp); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}

		switch statusResp.Status {
		case "transcribed":
			// Get transcript
			return m.getTranscript(ctx, jobID, statusResp.DurationSeconds, statusResp.Language)
		case "failed":
			return nil, fmt.Errorf("transcription failed")
		}
	}
}

func (m *RevAITranscriptionModel) getTranscript(ctx context.Context, jobID string, duration float64, language string) (*model.TranscriptionResult, error) {
	url := fmt.Sprintf("%s/jobs/%s/transcript", m.provider.opts.BaseURL, jobID)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	req.Header.Set("Accept", "application/vnd.rev.transcript.v1.0+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var transResp struct {
		Monologues []struct {
			Elements []struct {
				Type       string  `json:"type"`
				Value      string  `json:"value"`
				Timestamp  float64 `json:"ts"`
				EndTS      float64 `json:"end_ts"`
				Confidence float64 `json:"confidence"`
			} `json:"elements"`
		} `json:"monologues"`
	}
	if err := json.Unmarshal(body, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Build full text and segments
	var fullText string
	var segments []model.TranscriptionSegment
	segmentID := 0

	for _, mono := range transResp.Monologues {
		for _, elem := range mono.Elements {
			if elem.Type == "text" {
				fullText += elem.Value
				segments = append(segments, model.TranscriptionSegment{
					ID:         segmentID,
					Text:       elem.Value,
					Start:      elem.Timestamp,
					End:        elem.EndTS,
					Confidence: elem.Confidence,
				})
				segmentID++
			} else if elem.Type == "punct" {
				fullText += elem.Value
			}
		}
	}

	return &model.TranscriptionResult{
		Text:     fullText,
		Segments: segments,
		Language: language,
		Duration: &duration,
		Usage: &model.TranscriptionUsage{
			DurationSeconds: duration,
		},
		Response: model.TranscriptionResponse{
			ID:    jobID,
			Model: m.id,
		},
	}, nil
}
