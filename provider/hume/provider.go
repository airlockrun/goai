// Package hume provides a Hume AI provider implementation for expressive speech.
package hume

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
	defaultBaseURL = "https://api.hume.ai/v0/tts"
)

// Options contains configuration for the Hume provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Hume provider.
type Provider struct {
	opts Options
}

// New creates a new Hume provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "hume" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &HumeSpeechModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		"octave",
		"octave-v1",
	}
}

var _ provider.Provider = (*Provider)(nil)

// HumeSpeechModel implements the SpeechModel interface.
type HumeSpeechModel struct {
	id       string
	provider *Provider
}

func (m *HumeSpeechModel) ID() string       { return m.id }
func (m *HumeSpeechModel) Provider() string { return "hume" }

func (m *HumeSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	reqBody := map[string]any{
		"text": opts.Text,
	}

	// Voice configuration
	if opts.Voice != "" {
		reqBody["voice"] = map[string]any{
			"name": opts.Voice,
		}
	}

	// Add provider options (e.g., emotions, style)
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := m.provider.opts.BaseURL + "/synthesize"

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Hume-Api-Key", m.provider.opts.APIKey)
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Hume API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Read audio data
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	mimeType := resp.Header.Get("Content-Type")
	// Default to audio/wav when Content-Type is empty or is the generic
	// "application/octet-stream" (which Go's http server auto-sets for binary data)
	if mimeType == "" || mimeType == "application/octet-stream" {
		mimeType = "audio/wav"
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
