package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk packages/openai/src/chat/openai-chat-language-model.test.ts
// logprobs tests.

func TestOpenAIChat_Logprobs_Request(t *testing.T) {
	t.Run("logprobs=true sends {logprobs:true}", func(t *testing.T) {
		body := captureOpenAIChatRequest(t, map[string]any{"logprobs": true})
		if body["logprobs"] != true {
			t.Errorf("logprobs = %v, want true", body["logprobs"])
		}
		if _, has := body["top_logprobs"]; has {
			t.Errorf("top_logprobs should be absent when logprobs=true, got %v", body["top_logprobs"])
		}
	})

	t.Run("logprobs=N sends {logprobs:true, top_logprobs:N}", func(t *testing.T) {
		body := captureOpenAIChatRequest(t, map[string]any{"logprobs": 5})
		if body["logprobs"] != true {
			t.Errorf("logprobs = %v, want true", body["logprobs"])
		}
		if body["top_logprobs"].(float64) != 5 {
			t.Errorf("top_logprobs = %v, want 5", body["top_logprobs"])
		}
	})

	t.Run("logprobs unset sends no logprobs field", func(t *testing.T) {
		body := captureOpenAIChatRequest(t, nil)
		if _, has := body["logprobs"]; has {
			t.Errorf("logprobs should be absent, got %v", body["logprobs"])
		}
	})
}

func TestOpenAIChat_Logprobs_ResponseMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.001,"top_logprobs":[{"token":"Hi","logprob":-0.001},{"token":"Hey","logprob":-2.5}]}]}}]}` + "\n\n"))
		w.Write([]byte(`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	m := createTestProvider(server.URL).Chat("gpt-4o-mini")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:        []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: map[string]any{"logprobs": true},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var finishMeta map[string]any
	for ev := range events {
		if ev.Type == stream.EventFinish {
			finishMeta = ev.Data.(stream.FinishEvent).ProviderMetadata
		}
	}
	if finishMeta == nil {
		t.Fatal("ProviderMetadata was nil")
	}
	lp := finishMeta["openai"].(map[string]any)["logprobs"].([]map[string]any)
	if len(lp) != 1 {
		t.Fatalf("logprobs len = %d, want 1", len(lp))
	}
	if lp[0]["token"] != "Hi" {
		t.Errorf("token = %v", lp[0]["token"])
	}
	if lp[0]["logprob"].(float64) != -0.001 {
		t.Errorf("logprob = %v", lp[0]["logprob"])
	}
	top := lp[0]["topLogprobs"].([]map[string]any)
	if len(top) != 2 {
		t.Fatalf("topLogprobs len = %d, want 2", len(top))
	}
	if top[1]["token"] != "Hey" {
		t.Errorf("top[1].token = %v", top[1]["token"])
	}
}

// captureOpenAIChatRequest drives a single streaming chat turn against a mock
// server and returns the parsed request body.
func captureOpenAIChatRequest(t *testing.T, providerOpts map[string]any) map[string]any {
	t.Helper()
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &received)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	m := createTestProvider(server.URL).Chat("gpt-4o-mini")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:        []message.Message{message.NewUserMessage("hi")},
		ProviderOptions: providerOpts,
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range events {
	}
	return received
}
