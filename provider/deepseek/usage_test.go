package deepseek

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Usage numbers mirror ai-sdk's deepseek-tool-call.json fixture: a
// deepseek-reasoner response reporting both prompt_cache_hit_tokens and the
// standard prompt_tokens_details.cached_tokens, plus reasoning tokens.
func TestDeepSeekModel_CacheAndReasoningUsageOnWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":339,"completion_tokens":83,"total_tokens":422,"prompt_tokens_details":{"cached_tokens":320},"completion_tokens_details":{"reasoning_tokens":39},"prompt_cache_hit_tokens":320,"prompt_cache_miss_tokens":19}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.Model("deepseek-reasoner")

	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var usage stream.Usage
	for ev := range events {
		if ev.Type == stream.EventFinish {
			usage = ev.Data.(stream.FinishEvent).Usage
		}
	}

	want := map[string]int{
		"in.total":   339,
		"in.noCache": 19,
		"in.cache":   320,
		"out.total":  83,
		"out.text":   44, // 83 - 39
		"out.reason": 39,
	}
	got := map[string]int{
		"in.total":   deref(usage.InputTokens.Total),
		"in.noCache": deref(usage.InputTokens.NoCache),
		"in.cache":   deref(usage.InputTokens.CacheRead),
		"out.total":  deref(usage.OutputTokens.Total),
		"out.text":   deref(usage.OutputTokens.Text),
		"out.reason": deref(usage.OutputTokens.Reasoning),
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("%s = %d, want %d", k, got[k], w)
		}
	}
}

func deref(p *int) int {
	if p == nil {
		return -1
	}
	return *p
}
