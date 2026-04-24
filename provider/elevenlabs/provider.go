// Package elevenlabs provides an ElevenLabs provider implementation for speech generation.
package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

const (
	defaultBaseURL = "https://api.elevenlabs.io/v1"
)

// Options contains configuration for the ElevenLabs provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the ElevenLabs provider.
type Provider struct {
	opts Options
}

// New creates a new ElevenLabs provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "elevenlabs" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &ElevenLabsSpeechModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		"eleven_multilingual_v2",
		"eleven_turbo_v2_5",
		"eleven_turbo_v2",
		"eleven_monolingual_v1",
	}
}

var _ provider.Provider = (*Provider)(nil)

// ElevenLabsSpeechModel implements the SpeechModel interface.
type ElevenLabsSpeechModel struct {
	id       string
	provider *Provider
}

func (m *ElevenLabsSpeechModel) ID() string       { return m.id }
func (m *ElevenLabsSpeechModel) Provider() string { return "elevenlabs" }

func (m *ElevenLabsSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	// Voice ID is required
	voiceID := opts.Voice
	if voiceID == "" {
		voiceID = "21m00Tcm4TlvDq8ikWAM" // Default Rachel voice
	}

	// Build request
	req := speechRequest{
		Text:    opts.Text,
		ModelID: m.id,
	}

	// Voice settings from provider options
	if settings, ok := opts.ProviderOptions["voice_settings"].(map[string]any); ok {
		req.VoiceSettings = settings
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/text-to-speech/%s", m.provider.opts.BaseURL, voiceID)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("xi-api-key", m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	// Accept header for audio format
	outputFormat := opts.OutputFormat
	if outputFormat == "" {
		outputFormat = "mp3_44100_128"
	}
	httpReq.Header.Set("Accept", "audio/mpeg")

	// Execute request
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ElevenLabs API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Read audio data
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	return &model.SpeechResult{
		Audio:    audio,
		MimeType: "audio/mpeg",
		Usage: &model.SpeechUsage{
			Characters: len(opts.Text),
		},
		Response: model.SpeechResponse{
			Model: m.id,
		},
	}, nil
}

type speechRequest struct {
	Text          string         `json:"text"`
	ModelID       string         `json:"model_id"`
	VoiceSettings map[string]any `json:"voice_settings,omitempty"`
}
