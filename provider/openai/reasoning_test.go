package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// CallOptions.Reasoning lowers into wire reasoning_effort on Chat
// Completions and reasoning.effort on Responses. Provider-specific
// opts.ReasoningEffort takes precedence when both are set. Mirrors
// ai-sdk v4 reasoning enum.

// captureModelRequestBody is a thin helper that drives a Chat or
// Responses model against a stub server and returns the request body.
func captureModelRequestBody(t *testing.T, modelID string, useResponses bool, callOpts *stream.CallOptions) map[string]any {
	t.Helper()
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if useResponses {
			// Minimal Responses-API stream.
			w.Write([]byte("data: " + `{"type":"response.created","response":{"id":"r","model":"x","object":"response"}}` + "\n\n"))
			w.Write([]byte("data: " + `{"type":"response.completed","response":{"id":"r","model":"x","object":"response","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` + "\n\n"))
		} else {
			w.Write([]byte("data: " + `{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}` + "\n\n"))
			w.Write([]byte("data: " + `{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: server.URL})
	var m stream.Model
	if useResponses {
		m = p.Responses(modelID)
	} else {
		m = p.Chat(modelID)
	}
	events, err := m.Stream(context.Background(), callOpts)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	return captured
}

func TestChatModel_ReasoningLowersToWire(t *testing.T) {
	body := captureModelRequestBody(t, "gpt-5", false, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortMedium,
	})
	if body["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", body["reasoning_effort"])
	}
}

func TestChatModel_ProviderEffortWinsOverReasoning(t *testing.T) {
	body := captureModelRequestBody(t, "gpt-5", false, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortLow,
		ProviderOptions: map[string]any{
			"reasoningEffort": "high",
		},
	})
	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high (provider option overrides Reasoning)", body["reasoning_effort"])
	}
}

func TestChatModel_ReasoningOmittedOnNonReasoningModel(t *testing.T) {
	body := captureModelRequestBody(t, "gpt-4o-mini", false, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortMedium,
	})
	if _, has := body["reasoning_effort"]; has {
		t.Errorf("reasoning_effort should be omitted on non-reasoning model, got %v", body["reasoning_effort"])
	}
}

func TestResponsesModel_ReasoningLowersToWire(t *testing.T) {
	body := captureModelRequestBody(t, "gpt-5", true, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortMedium,
	})
	r, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("expected reasoning object, got %T (%v)", body["reasoning"], body["reasoning"])
	}
	if r["effort"] != "medium" {
		t.Errorf("reasoning.effort = %v, want medium", r["effort"])
	}
}

func TestResponsesModel_ProviderEffortWinsOverReasoning(t *testing.T) {
	body := captureModelRequestBody(t, "gpt-5", true, &stream.CallOptions{
		Messages:  []message.Message{message.NewUserMessage("hi")},
		Reasoning: stream.ReasoningEffortLow,
		ProviderOptions: map[string]any{
			"reasoningEffort": "high",
		},
	})
	r, _ := body["reasoning"].(map[string]any)
	if r["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want high (provider option overrides Reasoning)", r["effort"])
	}
}
