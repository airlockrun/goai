// Package gladia provides a Gladia provider implementation for transcription.
package gladia

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.gladia.io/v2"
)

// Options contains configuration for the Gladia provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Gladia provider.
type Provider struct {
	opts Options
}

// New creates a new Gladia provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "gladia" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel       { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &GladiaTranscriptionModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) Models() []string {
	return []string{
		"default",
		"enhanced",
	}
}

var _ provider.Provider = (*Provider)(nil)

// GladiaTranscriptionModel implements the TranscriptionModel interface.
type GladiaTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *GladiaTranscriptionModel) ID() string       { return m.id }
func (m *GladiaTranscriptionModel) Provider() string { return "gladia" }

func (m *GladiaTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
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

	// Upload audio first
	uploadURL, err := m.uploadAudio(ctx, audioData, opts.MimeType)
	if err != nil {
		return nil, err
	}

	return m.transcribeFromURL(ctx, uploadURL, opts)
}

func (m *GladiaTranscriptionModel) uploadAudio(ctx context.Context, audioData []byte, mimeType string) (string, error) {
	url := fmt.Sprintf("%s/upload", m.provider.opts.BaseURL)

	if mimeType == "" {
		mimeType = "audio/wav"
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(audioData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", mimeType)
	req.Header.Set("x-gladia-key", m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("Gladia upload error (status %d): %s", resp.StatusCode, string(body))
	}

	var uploadResp struct {
		AudioURL string `json:"audio_url"`
	}
	if err := json.Unmarshal(body, &uploadResp); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %w", err)
	}

	return uploadResp.AudioURL, nil
}

func (m *GladiaTranscriptionModel) transcribeFromURL(ctx context.Context, audioURL string, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	reqBody := map[string]any{
		"audio_url": audioURL,
	}

	if opts.Language != "" {
		reqBody["language"] = opts.Language
	} else {
		reqBody["detect_language"] = true
	}

	// Enable word-level timestamps
	reqBody["subtitles"] = true

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/transcription", m.provider.opts.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-gladia-key", m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("Gladia API error (status %d): %s", resp.StatusCode, string(body))
	}

	var initResp struct {
		ID        string `json:"id"`
		ResultURL string `json:"result_url"`
	}
	if err := json.Unmarshal(body, &initResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for result
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		result, done, err := m.getResult(ctx, initResp.ResultURL)
		if err != nil {
			return nil, err
		}
		if done {
			result.Response.ID = initResp.ID
			return result, nil
		}
	}
}

func (m *GladiaTranscriptionModel) getResult(ctx context.Context, resultURL string) (*model.TranscriptionResult, bool, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", resultURL, nil)
	if err != nil {
		return nil, false, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("x-gladia-key", m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, false, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, fmt.Errorf("failed to read response: %w", err)
	}

	var statusResp struct {
		Status string `json:"status"`
		Result struct {
			Transcription struct {
				FullTranscript string `json:"full_transcript"`
				Languages      []struct {
					Language string `json:"language"`
				} `json:"languages"`
				Utterances []struct {
					Text       string  `json:"text"`
					Start      float64 `json:"start"`
					End        float64 `json:"end"`
					Confidence float64 `json:"confidence"`
					Words      []struct {
						Word       string  `json:"word"`
						Start      float64 `json:"start"`
						End        float64 `json:"end"`
						Confidence float64 `json:"confidence"`
					} `json:"words"`
				} `json:"utterances"`
			} `json:"transcription"`
			Metadata struct {
				AudioDuration float64 `json:"audio_duration"`
			} `json:"metadata"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &statusResp); err != nil {
		return nil, false, fmt.Errorf("failed to parse response: %w", err)
	}

	switch statusResp.Status {
	case "done":
		// Convert utterances to segments
		segments := make([]model.TranscriptionSegment, len(statusResp.Result.Transcription.Utterances))
		for i, utt := range statusResp.Result.Transcription.Utterances {
			words := make([]model.TranscriptionWord, len(utt.Words))
			for j, w := range utt.Words {
				words[j] = model.TranscriptionWord{
					Word:       w.Word,
					Start:      w.Start,
					End:        w.End,
					Confidence: w.Confidence,
				}
			}
			segments[i] = model.TranscriptionSegment{
				ID:         i,
				Text:       utt.Text,
				Start:      utt.Start,
				End:        utt.End,
				Confidence: utt.Confidence,
				Words:      words,
			}
		}

		language := ""
		if len(statusResp.Result.Transcription.Languages) > 0 {
			language = statusResp.Result.Transcription.Languages[0].Language
		}

		duration := statusResp.Result.Metadata.AudioDuration

		return &model.TranscriptionResult{
			Text:     statusResp.Result.Transcription.FullTranscript,
			Segments: segments,
			Language: language,
			Duration: &duration,
			Usage: &model.TranscriptionUsage{
				DurationSeconds: duration,
			},
			Response: model.TranscriptionResponse{
				Model: m.id,
			},
		}, true, nil
	case "error":
		return nil, false, fmt.Errorf("transcription failed")
	default:
		// Still processing
		return nil, false, nil
	}
}
