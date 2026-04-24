// Package lmnt provides an LMNT provider implementation for speech generation.
package lmnt

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
	defaultBaseURL = "https://api.lmnt.com/v1"
)

// Options contains configuration for the LMNT provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the LMNT provider.
type Provider struct {
	opts Options
}

// New creates a new LMNT provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                         { return "lmnt" }
func (p *Provider) Model(modelID string) stream.Model                  { return nil }
func (p *Provider) LanguageModel(modelID string) model.LanguageModel   { return nil }
func (p *Provider) ImageModel(modelID string) model.ImageModel         { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel {
	return &LMNTSpeechModel{
		id:       modelID,
		provider: p,
	}
}
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		"lily",
		"daniel",
		"mrf-english",
	}
}

var _ provider.Provider = (*Provider)(nil)

// LMNTSpeechModel implements the SpeechModel interface.
type LMNTSpeechModel struct {
	id       string
	provider *Provider
}

func (m *LMNTSpeechModel) ID() string       { return m.id }
func (m *LMNTSpeechModel) Provider() string { return "lmnt" }

func (m *LMNTSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	// Build request
	reqBody := map[string]any{
		"text":  opts.Text,
		"voice": m.id,
	}

	// Output format
	outputFormat := opts.OutputFormat
	if outputFormat == "" {
		outputFormat = "mp3"
	}
	reqBody["format"] = outputFormat

	// Speed
	if opts.Speed != nil {
		reqBody["speed"] = *opts.Speed
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/ai/speech", m.provider.opts.BaseURL)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-API-Key", m.provider.opts.APIKey)
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
		return nil, fmt.Errorf("LMNT API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Read audio data
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	mimeType := "audio/mpeg"
	switch outputFormat {
	case "wav":
		mimeType = "audio/wav"
	case "aac":
		mimeType = "audio/aac"
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
