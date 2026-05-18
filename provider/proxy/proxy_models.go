// Proxy implementations for non-language models (image, embedding, speech, transcription).
// These use simple JSON request/response (not NDJSON streaming).
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/airlockrun/goai/model"
)

// modelProxyRequest is the JSON body sent to non-streaming model endpoints.
type modelProxyRequest struct {
	Slug       string          `json:"slug,omitempty"`
	Capability string          `json:"capability"`
	Options    json.RawMessage `json:"options"`
}

// --- Image Model ---

// ImageModel returns a model.ImageModel that proxies calls through Airlock.
func ImageModel(opts Options) model.ImageModel {
	if opts.Path == "" {
		opts.Path = "/api/agent/llm/image"
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxRetries == 0 {
		opts.MaxRetries = 5
	}
	return &proxyImageModel{opts: opts}
}

type proxyImageModel struct {
	opts Options
}

func (m *proxyImageModel) ID() string             { return "proxy" }
func (m *proxyImageModel) Provider() string        { return "proxy" }
func (m *proxyImageModel) MaxImagesPerCall() int   { return 4 }

func (m *proxyImageModel) Generate(ctx context.Context, callOpts model.ImageCallOptions) (*model.ImageResult, error) {
	var result model.ImageResult
	if err := doModelProxy(ctx, m.opts, "image", callOpts, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- Embedding Model ---

// EmbeddingModel returns a model.EmbeddingModel that proxies calls through Airlock.
func EmbeddingModel(opts Options) model.EmbeddingModel {
	if opts.Path == "" {
		opts.Path = "/api/agent/llm/embedding"
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxRetries == 0 {
		opts.MaxRetries = 5
	}
	return &proxyEmbeddingModel{opts: opts}
}

type proxyEmbeddingModel struct {
	opts Options
}

func (m *proxyEmbeddingModel) ID() string                { return "proxy" }
func (m *proxyEmbeddingModel) Provider() string           { return "proxy" }
func (m *proxyEmbeddingModel) MaxEmbeddingsPerCall() int  { return 100 }
func (m *proxyEmbeddingModel) Dimensions() int            { return 0 }

func (m *proxyEmbeddingModel) Embed(ctx context.Context, callOpts model.EmbedCallOptions) (*model.EmbedResult, error) {
	var result model.EmbedResult
	if err := doModelProxy(ctx, m.opts, "embedding", callOpts, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- Speech Model ---

// SpeechModel returns a model.SpeechModel that proxies calls through Airlock.
func SpeechModel(opts Options) model.SpeechModel {
	if opts.Path == "" {
		opts.Path = "/api/agent/llm/speech"
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxRetries == 0 {
		opts.MaxRetries = 5
	}
	return &proxySpeechModel{opts: opts}
}

type proxySpeechModel struct {
	opts Options
}

func (m *proxySpeechModel) ID() string       { return "proxy" }
func (m *proxySpeechModel) Provider() string  { return "proxy" }

func (m *proxySpeechModel) Generate(ctx context.Context, callOpts model.SpeechCallOptions) (*model.SpeechResult, error) {
	var result model.SpeechResult
	if err := doModelProxy(ctx, m.opts, "speech", callOpts, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- Transcription Model ---

// TranscriptionModel returns a model.TranscriptionModel that proxies calls through Airlock.
func TranscriptionModel(opts Options) model.TranscriptionModel {
	if opts.Path == "" {
		opts.Path = "/api/agent/llm/transcription"
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxRetries == 0 {
		opts.MaxRetries = 5
	}
	return &proxyTranscriptionModel{opts: opts}
}

type proxyTranscriptionModel struct {
	opts Options
}

func (m *proxyTranscriptionModel) ID() string       { return "proxy" }
func (m *proxyTranscriptionModel) Provider() string  { return "proxy" }

func (m *proxyTranscriptionModel) Transcribe(ctx context.Context, callOpts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	var result model.TranscriptionResult
	if err := doModelProxy(ctx, m.opts, "transcription", callOpts, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- Shared proxy logic ---

// doModelProxy sends a JSON request to the proxy endpoint and decodes the JSON response.
func doModelProxy(ctx context.Context, opts Options, capability string, callOpts any, result any) error {
	optsJSON, err := json.Marshal(callOpts)
	if err != nil {
		return fmt.Errorf("proxy: marshal options: %w", err)
	}

	body, err := json.Marshal(modelProxyRequest{
		Slug:       opts.Slug,
		Capability: capability,
		Options:    optsJSON,
	})
	if err != nil {
		return fmt.Errorf("proxy: marshal request: %w", err)
	}

	url := opts.BaseURL + opts.Path

	for attempt := 0; attempt <= opts.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("proxy: create request: %w", err)
		}
		applyHeaders(req, opts)

		resp, err := opts.Client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if attempt < opts.MaxRetries {
				sleepBackoff(ctx, attempt, nil)
				continue
			}
			return fmt.Errorf("proxy: request failed: %w", err)
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable {
			resp.Body.Close()
			if attempt < opts.MaxRetries {
				sleepBackoff(ctx, attempt, resp)
				continue
			}
			return fmt.Errorf("proxy: model returned %d after %d retries", resp.StatusCode, opts.MaxRetries)
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return fmt.Errorf("proxy: model returned status %d: %s", resp.StatusCode, string(respBody))
		}

		defer resp.Body.Close()
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("proxy: decode response: %w", err)
		}
		return nil
	}

	return fmt.Errorf("proxy: exhausted retries")
}
