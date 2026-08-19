package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/airlockrun/goai"
	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func TestStreamRetriesRetryableSetupStatus(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "upstream unavailable", http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"type\":\"text-delta\",\"data\":{\"text\":\"ok\"}}\n{\"type\":\"finish\",\"data\":{\"finishReason\":\"stop\"}}\n"))
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, Messages: []message.Message{message.NewUserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for event := range result.FullStream {
		if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
			text += delta.Text
		}
		if event.Type == stream.EventError {
			t.Fatalf("unexpected stream error: %v", event.Data)
		}
	}
	if text != "ok" {
		t.Fatalf("text = %q, want ok", text)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2", got)
	}
}

func TestStreamDoesNotRetryBodyError(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"type\":\"error\",\"data\":{\"error\":\"connection reset\"}}\n"))
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	events, err := model.Stream(context.Background(), &stream.CallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var sawError bool
	for event := range events {
		sawError = sawError || event.Type == stream.EventError
	}
	if !sawError {
		t.Fatal("expected body stream error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamTextExplicitZeroDisablesProxySetupRetries(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, MaxRetries: 0, MaxRetriesSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.Text(); err == nil {
		t.Fatal("Text() error = nil, want setup error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamTextHonorsNonRetryableAirlockSetupError(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("X-Airlock-LLM-Retryable", "false")
		http.Error(w, "invalid provider options", http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.Text(); err == nil {
		t.Fatal("Text() error = nil, want setup error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStreamBoundsSetupErrorBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, strings.Repeat("x", maxErrorResponseBodyBytes*2), http.StatusBadGateway)
	}))
	defer server.Close()

	model := Model("", Options{BaseURL: server.URL})
	result, err := goai.StreamText(context.Background(), stream.Input{
		Model: model, MaxRetriesSet: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = result.Text()
	var apiErr *goaierrors.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Text() error = %v, want APICallError", err)
	}
	if len(apiErr.ResponseBody) > maxErrorResponseBodyBytes+len("... (truncated)") {
		t.Fatalf("response body length = %d, want bounded", len(apiErr.ResponseBody))
	}
	if !strings.HasSuffix(apiErr.ResponseBody, "... (truncated)") {
		t.Fatalf("response body was not marked truncated: %q", apiErr.ResponseBody[len(apiErr.ResponseBody)-32:])
	}
}
