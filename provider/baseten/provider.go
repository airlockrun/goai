// Package baseten provides a Baseten provider implementation.
package baseten

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
	defaultBaseURL = "https://model-%s.api.baseten.co/production/predict"
)

// Options contains configuration for the Baseten provider.
type Options struct {
	APIKey  string
	Headers map[string]string
}

// Provider implements the Baseten provider.
type Provider struct {
	opts Options
}

// New creates a new Baseten provider.
func New(opts Options) *Provider {
	return &Provider{opts: opts}
}

func (p *Provider) ID() string { return "baseten" }

func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return &BasetenLanguageModel{
		id:       modelID,
		provider: p,
	}
}

func (p *Provider) ImageModel(modelID string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(modelID string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(modelID string) model.RerankingModel         { return nil }

func (p *Provider) Models() []string {
	return []string{
		// Model IDs are deployment-specific
	}
}

var _ provider.Provider = (*Provider)(nil)

// BasetenLanguageModel implements the LanguageModel interface.
type BasetenLanguageModel struct {
	id       string
	provider *Provider
}

func (m *BasetenLanguageModel) ID() string       { return m.id }
func (m *BasetenLanguageModel) Provider() string { return "baseten" }

func (m *BasetenLanguageModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)

	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()

	return events, nil
}

func (m *BasetenLanguageModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	// Baseten exposes a plain prompt/completion API with no structured-output
	// endpoint. When ResponseFormat asks for JSON, inject a JSON instruction
	// so the model at least returns JSON-formatted text.
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	// Build prompt from messages (Baseten models typically use prompt-based input)
	var prompt strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case message.RoleSystem:
			prompt.WriteString("<|system|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n")
		case message.RoleUser:
			prompt.WriteString("<|user|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n")
		case message.RoleAssistant:
			prompt.WriteString("<|assistant|>\n")
			prompt.WriteString(msg.Content.Text)
			prompt.WriteString("\n")
		}
	}
	prompt.WriteString("<|assistant|>\n")

	reqBody := map[string]any{
		"prompt": prompt.String(),
		"stream": true,
	}

	if options.MaxOutputTokens != nil {
		reqBody["max_tokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		reqBody["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		reqBody["top_p"] = *options.TopP
	}
	if options.TopK != nil {
		reqBody["top_k"] = *options.TopK
	}
	if len(options.StopSequences) > 0 {
		reqBody["stop"] = options.StopSequences
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := fmt.Sprintf(defaultBaseURL, m.id)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBytes))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Api-Key "+m.provider.opts.APIKey)
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
			Data: stream.ErrorEvent{Error: fmt.Errorf("Baseten API error (status %d): %s", resp.StatusCode, string(body))},
		}
		return
	}

	m.processStream(ctx, resp.Body, options.Tools, events)
}

func (m *BasetenLanguageModel) processStream(ctx context.Context, body io.Reader, tools []tool.Tool, events chan<- stream.Event) {
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

		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Token string `json:"token"`
			Text  string `json:"text"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		text := chunk.Token
		if text == "" {
			text = chunk.Text
		}

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
