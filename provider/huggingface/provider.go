// Package huggingface provides a Hugging Face Inference API provider implementation.
package huggingface

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

const (
	defaultBaseURL = "https://api-inference.huggingface.co/models"
)

// Options contains configuration for the Hugging Face provider.
type Options struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

// Provider implements the Hugging Face provider.
type Provider struct {
	opts Options
}

// New creates a new Hugging Face provider.
func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string { return "huggingface" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return &HuggingFaceLanguageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel {
	return &HuggingFaceImageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel {
	return &HuggingFaceEmbeddingModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		"meta-llama/Meta-Llama-3.1-8B-Instruct",
		"meta-llama/Meta-Llama-3.1-70B-Instruct",
		"mistralai/Mistral-7B-Instruct-v0.3",
		"mistralai/Mixtral-8x7B-Instruct-v0.1",
		"microsoft/Phi-3-mini-4k-instruct",
		"sentence-transformers/all-MiniLM-L6-v2",
		"black-forest-labs/FLUX.1-dev",
		"stabilityai/stable-diffusion-xl-base-1.0",
	}
}

var _ provider.Provider = (*Provider)(nil)

// HuggingFaceLanguageModel implements the LanguageModel interface.
type HuggingFaceLanguageModel struct {
	id       string
	provider *Provider
}

func (m *HuggingFaceLanguageModel) ID() string       { return m.id }
func (m *HuggingFaceLanguageModel) Provider() string { return "huggingface" }

func (m *HuggingFaceLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *HuggingFaceLanguageModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// HuggingFace's Inference API is prompt-in / text-out — there is no
	// structured-output endpoint. When ResponseFormat asks for JSON, inject
	// the JSON instruction into the system message so the model still gets
	// the signal (and schema, if any).
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	// Build prompt from messages
	var prompt strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			prompt.WriteString("<|system|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n</s>\n")
		case message.RoleUser:
			prompt.WriteString("<|user|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n</s>\n")
		case message.RoleAssistant:
			prompt.WriteString("<|assistant|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n</s>\n")
		}
	}
	prompt.WriteString("<|assistant|>\n")

	reqBody := map[string]any{
		"inputs": prompt.String(),
		"parameters": map[string]any{
			"return_full_text": false,
		},
		"stream": true,
	}

	if options.MaxOutputTokens != nil {
		params := reqBody["parameters"].(map[string]any)
		params["max_new_tokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		params := reqBody["parameters"].(map[string]any)
		params["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		params := reqBody["parameters"].(map[string]any)
		params["top_p"] = *options.TopP
	}
	if options.TopK != nil {
		params := reqBody["parameters"].(map[string]any)
		params["top_k"] = *options.TopK
	}
	if len(options.StopSequences) > 0 {
		params := reqBody["parameters"].(map[string]any)
		params["stop"] = options.StopSequences
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf("%s/%s", m.provider.opts.BaseURL, m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range options.Headers {
		req.Header.Set(k, v)
	}

	events <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{}}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		events <- stream.Event{
			Type: stream.EventError,
			Data: stream.ErrorEvent{Error: fmt.Errorf("Hugging Face API error (status %d): %s", resp.StatusCode, string(body))},
		}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func (m *HuggingFaceLanguageModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var textStarted bool

	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimPrefix(line, "data:")
		data = strings.TrimSpace(data)

		if data == "" {
			continue
		}

		var chunk struct {
			Token struct {
				Text string `json:"text"`
			} `json:"token"`
			GeneratedText string `json:"generated_text"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		text := chunk.Token.Text
		if text != "" {
			if !textStarted {
				textStarted = true
				events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
			}
			events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}}
		}
	}

	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}

	events <- stream.Event{
		Type: stream.EventFinishStep,
		Data: stream.FinishStepEvent{FinishReason: stream.FinishReasonStop},
	}

	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop},
	}
}

// HuggingFaceEmbeddingModel implements the EmbeddingModel interface.
type HuggingFaceEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *HuggingFaceEmbeddingModel) ID() string                { return m.id }
func (m *HuggingFaceEmbeddingModel) Provider() string          { return "huggingface" }
func (m *HuggingFaceEmbeddingModel) MaxEmbeddingsPerCall() int { return 100 }
func (m *HuggingFaceEmbeddingModel) Dimensions() int           { return 0 }

func (m *HuggingFaceEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	reqBody := map[string]any{
		"inputs": opts.Values,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/%s", m.provider.opts.BaseURL, m.id)

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

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Hugging Face API error (status %d): %s", resp.StatusCode, string(body))
	}

	var embResp [][]float64
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	embeddings := make([]model.Embedding, len(embResp))
	for i, emb := range embResp {
		embeddings[i] = model.Embedding{
			Values: emb,
			Index:  i,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Response: model.EmbeddingResponse{
			Model: m.id,
		},
	}, nil
}

// HuggingFaceImageModel implements the ImageModel interface.
type HuggingFaceImageModel struct {
	id       string
	provider *Provider
}

func (m *HuggingFaceImageModel) ID() string            { return m.id }
func (m *HuggingFaceImageModel) Provider() string      { return "huggingface" }
func (m *HuggingFaceImageModel) MaxImagesPerCall() int { return 1 }

func (m *HuggingFaceImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	reqBody := map[string]any{
		"inputs": opts.Prompt,
	}

	// Add provider options
	for k, v := range opts.ProviderOptions {
		reqBody[k] = v
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/%s", m.provider.opts.BaseURL, m.id)

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

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Hugging Face API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Response is raw image bytes
	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read image data: %w", err)
	}

	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = "image/png"
	}

	return &model.ImageResult{
		Images: []model.GeneratedImage{
			{
				Base64:   string(imageData), // Note: This is raw bytes, not base64
				MimeType: mimeType,
			},
		},
		Response: model.ImageResponse{
			Model: m.id,
		},
	}, nil
}
