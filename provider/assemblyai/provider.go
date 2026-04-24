// Package assemblyai provides an AssemblyAI provider implementation for transcription.
package assemblyai

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
	defaultBaseURL = "https://api.assemblyai.com/v2"
)

// Options contains configuration for the AssemblyAI provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the AssemblyAI provider.
type Provider struct {
	opts Options
}

// New creates a new AssemblyAI provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "assemblyai" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel       { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &AssemblyAITranscriptionModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) Models() []string {
	return []string{
		"best",
		"nano",
	}
}

var _ provider.Provider = (*Provider)(nil)

// AssemblyAITranscriptionModel implements the TranscriptionModel interface.
type AssemblyAITranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *AssemblyAITranscriptionModel) ID() string       { return m.id }
func (m *AssemblyAITranscriptionModel) Provider() string { return "assemblyai" }

func (m *AssemblyAITranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
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
		// Use URL directly
		return m.transcribeFromURL(ctx, opts.AudioURL, opts)
	} else {
		return nil, fmt.Errorf("audio data is required")
	}

	// Upload audio
	uploadURL, err := m.uploadAudio(ctx, audioData)
	if err != nil {
		return nil, err
	}

	return m.transcribeFromURL(ctx, uploadURL, opts)
}

func (m *AssemblyAITranscriptionModel) uploadAudio(ctx context.Context, audioData []byte) (string, error) {
	url := fmt.Sprintf("%s/upload", m.provider.opts.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(audioData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Authorization", m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("AssemblyAI upload error (status %d): %s", resp.StatusCode, string(body))
	}

	var uploadResp struct {
		UploadURL string `json:"upload_url"`
	}
	if err := json.Unmarshal(body, &uploadResp); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %w", err)
	}

	return uploadResp.UploadURL, nil
}

func (m *AssemblyAITranscriptionModel) transcribeFromURL(ctx context.Context, audioURL string, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	// Create transcription request
	reqBody := map[string]any{
		"audio_url": audioURL,
	}

	// Model selection
	if m.id == "nano" {
		reqBody["speech_model"] = "nano"
	}

	if opts.Language != "" {
		reqBody["language_code"] = opts.Language
	} else {
		reqBody["language_detection"] = true
	}

	// Enable speaker diarization if requested
	if speakerLabels, ok := opts.ProviderOptions["speaker_labels"].(bool); ok && speakerLabels {
		reqBody["speaker_labels"] = true
	}

	// Enable word timestamps
	reqBody["word_boost"] = []string{}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/transcript", m.provider.opts.BaseURL)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", m.provider.opts.APIKey)
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

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("AssemblyAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	var transResp transcriptResponse
	if err := json.Unmarshal(body, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Poll for completion
	for transResp.Status == "queued" || transResp.Status == "processing" {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		transResp, err = m.getTranscript(ctx, transResp.ID)
		if err != nil {
			return nil, err
		}
	}

	if transResp.Status == "error" {
		return nil, fmt.Errorf("transcription failed: %s", transResp.Error)
	}

	// Convert words to segments
	segments := make([]model.TranscriptionSegment, len(transResp.Words))
	for i, word := range transResp.Words {
		segments[i] = model.TranscriptionSegment{
			ID:         i,
			Text:       word.Text,
			Start:      float64(word.Start) / 1000.0,
			End:        float64(word.End) / 1000.0,
			Confidence: word.Confidence,
		}
	}

	var duration *float64
	if transResp.AudioDuration > 0 {
		d := float64(transResp.AudioDuration)
		duration = &d
	}

	return &model.TranscriptionResult{
		Text:     transResp.Text,
		Segments: segments,
		Language: transResp.Language,
		Duration: duration,
		Usage: &model.TranscriptionUsage{
			DurationSeconds: float64(transResp.AudioDuration),
		},
		Response: model.TranscriptionResponse{
			ID:    transResp.ID,
			Model: m.id,
		},
	}, nil
}

func (m *AssemblyAITranscriptionModel) getTranscript(ctx context.Context, id string) (transcriptResponse, error) {
	url := fmt.Sprintf("%s/transcript/%s", m.provider.opts.BaseURL, id)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return transcriptResponse{}, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", m.provider.opts.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return transcriptResponse{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return transcriptResponse{}, fmt.Errorf("failed to read response: %w", err)
	}

	var transResp transcriptResponse
	if err := json.Unmarshal(body, &transResp); err != nil {
		return transcriptResponse{}, fmt.Errorf("failed to parse response: %w", err)
	}

	return transResp, nil
}

type transcriptResponse struct {
	ID            string  `json:"id"`
	Status        string  `json:"status"`
	Text          string  `json:"text"`
	Language      string  `json:"language_code"`
	AudioDuration float64 `json:"audio_duration"`
	Error         string  `json:"error"`
	Words         []struct {
		Text       string  `json:"text"`
		Start      int     `json:"start"`
		End        int     `json:"end"`
		Confidence float64 `json:"confidence"`
	} `json:"words"`
}
