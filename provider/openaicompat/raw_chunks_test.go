package openaicompat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// CallOptions.IncludeRawChunks emits a RawChunkEvent for every parsed
// SSE payload before the translated events. Mirrors ai-sdk v4's
// includeRawChunks (LanguageModelV4CallOptions). Tested via openaicompat
// since the same pattern fans out to every other streaming provider in
// goai (anthropic/openai/google/etc.).

func newRawChunksServer() *httptest.Server {
	chunks := []string{
		`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
		`data: {"id":"x","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		`data: [DONE]`,
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, c := range chunks {
			w.Write([]byte(c + "\n\n"))
		}
	}))
}

func TestOpenAICompat_IncludeRawChunks_OnEmitsRawEvents(t *testing.T) {
	server := newRawChunksServer()
	defer server.Close()

	p := New(Options{ProviderID: "x", APIKey: "k", BaseURL: server.URL})
	m := p.Model("m")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:         []message.Message{message.NewUserMessage("hi")},
		IncludeRawChunks: true,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var raws []stream.RawChunkEvent
	var firstNonRawIndex = -1
	var sawTextStart bool
	idx := 0
	for ev := range events {
		if rc, ok := ev.Data.(stream.RawChunkEvent); ok {
			raws = append(raws, rc)
			idx++
			continue
		}
		if firstNonRawIndex == -1 {
			firstNonRawIndex = idx
		}
		if _, ok := ev.Data.(stream.TextStartEvent); ok {
			sawTextStart = true
		}
		idx++
	}

	// Three usable SSE payloads (skip [DONE]).
	if len(raws) != 3 {
		t.Errorf("expected 3 RawChunkEvent, got %d", len(raws))
	}
	if !sawTextStart {
		t.Error("expected normal events to keep flowing alongside raw chunks")
	}
	// Raw payloads carry the trimmed `data: ...` strings.
	for i, rc := range raws {
		s, ok := rc.RawValue.(string)
		if !ok || s == "" {
			t.Errorf("raw[%d].RawValue should be a non-empty string, got %T (%v)", i, rc.RawValue, rc.RawValue)
		}
	}
}

func TestOpenAICompat_IncludeRawChunks_DefaultOffEmitsNoRawEvents(t *testing.T) {
	server := newRawChunksServer()
	defer server.Close()

	p := New(Options{ProviderID: "x", APIKey: "k", BaseURL: server.URL})
	m := p.Model("m")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		// IncludeRawChunks left at default (false).
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var rawCount int
	for ev := range events {
		if _, ok := ev.Data.(stream.RawChunkEvent); ok {
			rawCount++
		}
	}
	if rawCount != 0 {
		t.Errorf("expected no RawChunkEvent when off, got %d", rawCount)
	}
}
