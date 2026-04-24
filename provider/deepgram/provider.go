// Package deepgram provides a Deepgram provider implementation for speech and transcription.
package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.deepgram.com/v1"
)

// Options contains configuration for the Deepgram provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Deepgram provider.
type Provider struct {
	opts Options
}

// New creates a new Deepgram provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "deepgram" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &DeepgramSpeechModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel {
	return &DeepgramTranscriptionModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

func (p *Provider) Models() []string {
	return []string{
		"nova-2",
		"nova",
		"enhanced",
		"base",
	}
}

func (p *Provider) SpeechModels() []string {
	return []string{
		"aura-asteria-en",
		"aura-luna-en",
		"aura-stella-en",
		"aura-athena-en",
	}
}

var _ provider.Provider = (*Provider)(nil)

// DeepgramSpeechModel implements the SpeechModel interface.
type DeepgramSpeechModel struct {
	id       string
	provider *Provider
}

func (m *DeepgramSpeechModel) ID() string       { return m.id }
func (m *DeepgramSpeechModel) Provider() string { return "deepgram" }

func (m *DeepgramSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	// Build URL with query params
	params := url.Values{}
	params.Set("model", m.id)

	reqURL := fmt.Sprintf("%s/speak?%s", m.provider.opts.BaseURL, params.Encode())

	// Build request body
	reqBody := map[string]string{
		"text": opts.Text,
	}
	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", reqURL, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Token "+m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("Deepgram API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Read audio data
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = "audio/mpeg"
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

// DeepgramTranscriptionModel implements the TranscriptionModel interface.
type DeepgramTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *DeepgramTranscriptionModel) ID() string       { return m.id }
func (m *DeepgramTranscriptionModel) Provider() string { return "deepgram" }

func (m *DeepgramTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
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

	// Build URL with query params
	params := url.Values{}
	params.Set("model", m.id)
	params.Set("punctuate", "true")
	params.Set("utterances", "true")

	if opts.Language != "" {
		params.Set("language", opts.Language)
	}

	reqURL := fmt.Sprintf("%s/listen?%s", m.provider.opts.BaseURL, params.Encode())

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", reqURL, bytes.NewReader(audioData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	contentType := opts.MimeType
	if contentType == "" {
		contentType = "audio/wav"
	}
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("Authorization", "Token "+m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("Deepgram API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var transResp transcriptionResponse
	if err := json.Unmarshal(body, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract text from results
	text := ""
	var segments []model.TranscriptionSegment
	if len(transResp.Results.Channels) > 0 && len(transResp.Results.Channels[0].Alternatives) > 0 {
		alt := transResp.Results.Channels[0].Alternatives[0]
		text = alt.Transcript

		// Convert words to segments
		for i, word := range alt.Words {
			segments = append(segments, model.TranscriptionSegment{
				ID:         i,
				Text:       word.Word,
				Start:      word.Start,
				End:        word.End,
				Confidence: word.Confidence,
			})
		}
	}

	var duration *float64
	if transResp.Metadata.Duration > 0 {
		duration = &transResp.Metadata.Duration
	}

	return &model.TranscriptionResult{
		Text:     text,
		Segments: segments,
		Duration: duration,
		Usage: &model.TranscriptionUsage{
			DurationSeconds: transResp.Metadata.Duration,
		},
		Response: model.TranscriptionResponse{
			ID:    transResp.Metadata.RequestID,
			Model: m.id,
		},
	}, nil
}

// Response types

type transcriptionResponse struct {
	Metadata struct {
		RequestID string  `json:"request_id"`
		Duration  float64 `json:"duration"`
	} `json:"metadata"`
	Results struct {
		Channels []struct {
			Alternatives []struct {
				Transcript string  `json:"transcript"`
				Confidence float64 `json:"confidence"`
				Words      []struct {
					Word       string  `json:"word"`
					Start      float64 `json:"start"`
					End        float64 `json:"end"`
					Confidence float64 `json:"confidence"`
				} `json:"words"`
			} `json:"alternatives"`
		} `json:"channels"`
	} `json:"results"`
}
