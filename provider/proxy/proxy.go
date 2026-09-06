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
	"io"
	"math"
	"net/http"
	"strconv"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	goaiinternal "github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/stream"
)

const maxErrorResponseBodyBytes = 8 << 10

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

	// MaxRetries overrides the default number of setup retries performed by
	// StreamText and GenerateText. Zero leaves the default unchanged.
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
func (m *proxyModel) Provider() string { return "proxy" }
func (m *proxyModel) SetupMaxRetries() (int, bool) {
	return m.opts.MaxRetries, m.opts.MaxRetries != 0
}

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

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
	applyHeaders(req, m.opts)

	resp, err := m.opts.Client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			err = ctx.Err()
		}
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "proxy: LLM request failed", URL: url, Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true,
		})}}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		responseBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorResponseBodyBytes+1))
		if len(responseBody) > maxErrorResponseBodyBytes {
			responseBody = append(responseBody[:maxErrorResponseBodyBytes], []byte("... (truncated)")...)
		}
		retryable, retryableSet := false, false
		if raw := resp.Header.Get("X-Airlock-LLM-Retryable"); raw != "" {
			if parsed, parseErr := strconv.ParseBool(raw); parseErr == nil {
				retryable, retryableSet = parsed, true
			}
		}
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
			Message: "proxy: LLM request failed", URL: url, StatusCode: resp.StatusCode,
			ResponseHeaders: flattenHeaders(resp.Header), ResponseBody: string(responseBody),
			IsRetryable: retryable, IsRetryableSet: retryableSet,
		})}}
		return
	}

	// Parse NDJSON response.
	streamReader := goaiinternal.NewStreamReader(resp.Body)
	scanner := bufio.NewScanner(streamReader)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		event, err := parseNDJSONEvent(line)
		if err != nil {
			events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
			return
		}
		events <- event
	}
	if err := streamReader.Err(ctx, scanner.Err()); err != nil {
		events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		return
	}
}

func flattenHeaders(headers http.Header) map[string]string {
	flattened := make(map[string]string, len(headers))
	for name := range headers {
		flattened[name] = headers.Get(name)
	}
	return flattened
}

// sleepBackoff is used by the proxy's non-streaming model endpoints, whose
// retry loop is separate from StreamText and GenerateText.
func sleepBackoff(ctx context.Context, attempt int, resp *http.Response) {
	delay := time.Duration(math.Pow(2, float64(attempt))) * 2 * time.Second
	if resp != nil {
		if retryAfter := resp.Header.Get("Retry-After"); retryAfter != "" {
			if seconds, err := strconv.Atoi(retryAfter); err == nil {
				delay = time.Duration(seconds) * time.Second
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
		var d stream.ToolErrorEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
	case stream.EventToolOutputDenied:
		var d stream.ToolOutputDeniedEvent
		json.Unmarshal(envelope.Data, &d)
		data = d
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
