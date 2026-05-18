// Package proxy implements a stream.Model that proxies LLM calls through
// an Airlock-compatible NDJSON endpoint (POST /api/agent/llm/stream).
//
// This allows Sol and agentsdk to call LLMs without holding API keys —
// credentials are managed server-side by the proxy.
package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"time"

	"github.com/airlockrun/goai/stream"
)

// Options configures the proxy provider.
type Options struct {
	// BaseURL is the proxy server base URL (e.g., "http://localhost:8080").
	BaseURL string

	// Token is the Bearer token for authentication.
	Token string

	// Path is the proxy endpoint path. Defaults to "/api/agent/llm/stream".
	Path string

	// Slug is a run-level label for logging/tracking (e.g., "ocr", "summarize").
	Slug string

	// Capability is the model capability (e.g., "text", "vision").
	Capability string

	// Client is the HTTP client to use. Defaults to http.DefaultClient.
	Client *http.Client

	// MaxRetries is the maximum number of retry attempts for transient errors.
	// Defaults to 5.
	MaxRetries int

	// Headers are extra HTTP headers attached to every proxied request
	// (streaming and non-streaming). The proxy package itself is generic;
	// callers (e.g. agentsdk) use this to carry run attribution such as
	// X-Airlock-Run-ID. Authorization/Content-Type are set separately and
	// are not overridable here.
	Headers map[string]string
}

// applyHeaders sets Content-Type, optional bearer auth, and any caller-supplied
// extra headers on req. Extra headers cannot clobber Content-Type/Authorization.
func applyHeaders(req *http.Request, opts Options) {
	req.Header.Set("Content-Type", "application/json")
	if opts.Token != "" {
		req.Header.Set("Authorization", "Bearer "+opts.Token)
	}
	for k, v := range opts.Headers {
		if k == "Content-Type" || k == "Authorization" {
			continue
		}
		req.Header.Set(k, v)
	}
}

// proxyRequest is the JSON body sent to the proxy endpoint.
type proxyRequest struct {
	ModelID    string          `json:"model_id,omitempty"`
	Slug       string          `json:"slug,omitempty"`
	Capability string          `json:"capability,omitempty"`
	Options    json.RawMessage `json:"options"`
}

// Model returns a stream.Model that proxies calls through the configured endpoint.
// The modelID is the full provider/model string (e.g., "anthropic/claude-sonnet-4-20250514").
func Model(modelID string, opts Options) stream.Model {
	if opts.Path == "" {
		opts.Path = "/api/agent/llm/stream"
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxRetries == 0 {
		opts.MaxRetries = 5
	}
	return &proxyModel{
		modelID: modelID,
		opts:    opts,
	}
}

type proxyModel struct {
	modelID string
	opts    Options
}

func (m *proxyModel) ID() string       { return m.modelID }
func (m *proxyModel) Provider() string  { return "proxy" }

func (m *proxyModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	events := make(chan stream.Event, 100)
	go func() {
		defer close(events)
		m.doStream(ctx, options, events)
	}()
	return events, nil
}

func (m *proxyModel) doStream(ctx context.Context, options *stream.CallOptions, events chan<- stream.Event) {
	optsJSON, err := json.Marshal(options)
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	body, err := json.Marshal(proxyRequest{
		ModelID:    m.modelID,
		Slug:       m.opts.Slug,
		Capability: m.opts.Capability,
		Options:    optsJSON,
	})
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}

	url := m.opts.BaseURL + m.opts.Path

	for attempt := 0; attempt <= m.opts.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
			return
		}
		applyHeaders(req, m.opts)

		resp, err := m.opts.Client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: ctx.Err()}}
				return
			}
			if attempt < m.opts.MaxRetries {
				sleepBackoff(ctx, attempt, nil)
				continue
			}
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
			return
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable {
			resp.Body.Close()
			if attempt < m.opts.MaxRetries {
				sleepBackoff(ctx, attempt, resp)
				continue
			}
			events <- stream.Event{
				Type: stream.EventError,
				Data: stream.ErrorEvent{Error: fmt.Errorf("proxy: LLM returned %d after %d retries", resp.StatusCode, m.opts.MaxRetries)},
			}
			return
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			resp.Body.Close()
			events <- stream.Event{
				Type: stream.EventError,
				Data: stream.ErrorEvent{Error: fmt.Errorf("proxy: LLM returned status %d", resp.StatusCode)},
			}
			return
		}

		// Parse NDJSON response.
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			event, err := parseNDJSONEvent(line)
			if err != nil {
				events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
				resp.Body.Close()
				return
			}
			events <- event
		}
		resp.Body.Close()
		if err := scanner.Err(); err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		}
		return
	}
}

// sleepBackoff sleeps for 2^attempt * 2 seconds, respecting Retry-After and context.
func sleepBackoff(ctx context.Context, attempt int, resp *http.Response) {
	delay := time.Duration(math.Pow(2, float64(attempt))) * 2 * time.Second
	if resp != nil {
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if secs, err := strconv.Atoi(ra); err == nil {
				delay = time.Duration(secs) * time.Second
			}
		}
	}
	select {
	case <-time.After(delay):
	case <-ctx.Done():
	}
}

// parseNDJSONEvent parses a single NDJSON line into a stream.Event.
func parseNDJSONEvent(line []byte) (stream.Event, error) {
	var envelope struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(line, &envelope); err != nil {
		return stream.Event{}, fmt.Errorf("proxy: parse event: %w", err)
	}

	eventType := stream.EventType(envelope.Type)
	var data stream.EventData

	switch eventType {
	case stream.EventStart:
		data = stream.StartEvent{}
	case stream.EventTextStart:
		var d stream.TextStartEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventTextDelta:
		var d stream.TextDeltaEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventTextEnd:
		var d stream.TextEndEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolInputStart:
		var d stream.ToolInputStartEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolInputDelta:
		var d stream.ToolInputDeltaEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolInputEnd:
		var d stream.ToolInputEndEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolCall:
		var d stream.ToolCallEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolResult:
		var d stream.ToolResultEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolError:
		var d struct {
			ToolCallID string          `json:"toolCallId"`
			ToolName   string          `json:"toolName"`
			Input      json.RawMessage `json:"input,omitempty"`
			Error      string          `json:"error"`
		}
		json.Unmarshal(envelope.Data, &d)
		data = stream.ToolErrorEvent{
			ToolCallID: d.ToolCallID,
			ToolName:   d.ToolName,
			Input:      d.Input,
			Error:      fmt.Errorf("%s", d.Error),
		}
	case stream.EventReasoningStart:
		var d stream.ReasoningStartEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventReasoningDelta:
		var d stream.ReasoningDeltaEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventReasoningEnd:
		var d stream.ReasoningEndEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventStartStep:
		data = stream.StartStepEvent{}
	case stream.EventFinishStep:
		var d stream.FinishStepEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventFinish:
		var d stream.FinishEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventError:
		var d struct {
			Error string `json:"error"`
		}
		json.Unmarshal(envelope.Data, &d)
		data = stream.ErrorEvent{Error: fmt.Errorf("%s", d.Error)}
	default:
		return stream.Event{}, fmt.Errorf("proxy: unknown event type %q", envelope.Type)
	}

	return stream.Event{Type: eventType, Data: data}, nil
}
