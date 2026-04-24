package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Tests translated from ai-sdk's google-prepare-tools.test.ts and
// google-generative-ai-language-model.test.ts.

func TestGoogleGroundingTools_Request(t *testing.T) {
	t.Run("gemini-2 emits bare googleSearch object", func(t *testing.T) {
		receivedBody := captureGeminiRequest(t, "gemini-2.0-flash-exp", &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools:    []tool.Tool{GoogleSearch()},
		})

		tools := mustArray(t, receivedBody, "tools")
		if len(tools) != 1 {
			t.Fatalf("tools len = %d, want 1", len(tools))
		}
		gs, ok := tools[0].(map[string]any)["googleSearch"]
		if !ok {
			t.Errorf("missing googleSearch entry: %v", tools[0])
		}
		if m, _ := gs.(map[string]any); len(m) != 0 {
			t.Errorf("googleSearch = %v, want empty object", gs)
		}
	})

	t.Run("gemini-1.5-flash emits googleSearchRetrieval with dynamic config", func(t *testing.T) {
		receivedBody := captureGeminiRequest(t, "gemini-1.5-flash", &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools:    []tool.Tool{GoogleSearchWith(GoogleSearchOptions{Mode: "MODE_DYNAMIC", DynamicThreshold: 0.5})},
		})

		tools := mustArray(t, receivedBody, "tools")
		gsr, ok := tools[0].(map[string]any)["googleSearchRetrieval"]
		if !ok {
			t.Fatalf("missing googleSearchRetrieval: %v", tools[0])
		}
		drc := gsr.(map[string]any)["dynamicRetrievalConfig"].(map[string]any)
		if drc["mode"] != "MODE_DYNAMIC" {
			t.Errorf("mode = %v, want MODE_DYNAMIC", drc["mode"])
		}
		if drc["dynamicThreshold"].(float64) != 0.5 {
			t.Errorf("dynamicThreshold = %v, want 0.5", drc["dynamicThreshold"])
		}
	})

	t.Run("gemini-2 emits googleMaps / enterpriseWebSearch / urlContext / codeExecution", func(t *testing.T) {
		receivedBody := captureGeminiRequest(t, "gemini-2.0-flash-exp", &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools: []tool.Tool{
				GoogleMaps(),
				EnterpriseWebSearch(),
				URLContext(),
				CodeExecution(),
			},
		})

		tools := mustArray(t, receivedBody, "tools")
		if len(tools) != 4 {
			t.Fatalf("tools len = %d, want 4", len(tools))
		}
		expectedKeys := []string{"googleMaps", "enterpriseWebSearch", "urlContext", "codeExecution"}
		for i, key := range expectedKeys {
			entry := tools[i].(map[string]any)
			if _, ok := entry[key]; !ok {
				t.Errorf("tools[%d] missing key %q: %v", i, key, entry)
			}
		}
	})

	t.Run("function tools and provider tools coexist in request", func(t *testing.T) {
		receivedBody := captureGeminiRequest(t, "gemini-2.0-flash-exp", &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools: []tool.Tool{
				GoogleSearch(),
				{
					Name:        "my_fn",
					Description: "A function tool",
					InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
				},
			},
		})

		tools := mustArray(t, receivedBody, "tools")
		if len(tools) != 2 {
			t.Fatalf("tools len = %d, want 2", len(tools))
		}
		// Provider-defined tools come first per prepareGeminiTools.
		if _, ok := tools[0].(map[string]any)["googleSearch"]; !ok {
			t.Errorf("tools[0] should be googleSearch, got %v", tools[0])
		}
		fd, ok := tools[1].(map[string]any)["functionDeclarations"].([]any)
		if !ok || len(fd) != 1 {
			t.Fatalf("tools[1] should be functionDeclarations: %v", tools[1])
		}
		if fd[0].(map[string]any)["name"] != "my_fn" {
			t.Errorf("function name = %v, want my_fn", fd[0].(map[string]any)["name"])
		}
	})
}

func TestGoogleGroundingMetadata_Response(t *testing.T) {
	t.Run("groundingMetadata surfaces via providerMetadata.google", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			// Single chunk carrying the final metadata.
			w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"Paris"}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["capital france"],"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}}],"groundingSupports":[{"groundingChunkIndices":[0],"confidenceScores":[0.95]}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}` + "\n\n"))
		}))
		defer server.Close()

		m := createTestProvider(server.URL).Model("gemini-2.0-flash-exp")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("capital of france?")},
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
			t.Fatal("no FinishEvent metadata")
		}
		google := finishMeta["google"].(map[string]any)
		gm := google["groundingMetadata"].(map[string]any)
		queries := gm["webSearchQueries"].([]string)
		if len(queries) != 1 || queries[0] != "capital france" {
			t.Errorf("webSearchQueries = %v", queries)
		}
		chunks := gm["groundingChunks"].([]map[string]any)
		if len(chunks) != 1 {
			t.Fatalf("groundingChunks len = %d, want 1", len(chunks))
		}
		web := chunks[0]["web"].(map[string]any)
		if web["uri"] != "https://example.com" {
			t.Errorf("chunk.web.uri = %v", web["uri"])
		}
		supports := gm["groundingSupports"].([]map[string]any)
		if len(supports) != 1 {
			t.Fatalf("groundingSupports len = %d", len(supports))
		}
	})

	// Regression for ai-sdk #e2a59ef: groundingMetadata can arrive on
	// an earlier chunk before the chunk carrying finishReason. The
	// parser must preserve it across chunks rather than only reading
	// from the final frame.
	t.Run("groundingMetadata arriving before finishReason is preserved", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			// Chunk 1: grounding metadata + first text, no finishReason yet.
			w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"Paris"}]},"groundingMetadata":{"webSearchQueries":["capital france"],"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}}]}}]}` + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Chunk 2: finishReason + usage, NO grounding metadata.
			w.Write([]byte(`data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}` + "\n\n"))
		}))
		defer server.Close()

		m := createTestProvider(server.URL).Model("gemini-2.0-flash-exp")
		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("capital of france?")},
		})
		if err != nil {
			t.Fatal(err)
		}

		var finishMeta map[string]any
		for ev := range events {
			if ev.Type == stream.EventFinish {
				finishMeta = ev.Data.(stream.FinishEvent).ProviderMetadata
			}
		}
		if finishMeta == nil {
			t.Fatal("no FinishEvent metadata — grounding dropped when arriving before finishReason")
		}
		google, _ := finishMeta["google"].(map[string]any)
		gm, _ := google["groundingMetadata"].(map[string]any)
		if gm == nil {
			t.Fatalf("groundingMetadata not preserved across chunks: %+v", finishMeta)
		}
		queries, _ := gm["webSearchQueries"].([]string)
		if len(queries) != 1 || queries[0] != "capital france" {
			t.Errorf("webSearchQueries = %v (lost the early-chunk grounding)", queries)
		}
	})
}

// --- test helpers ---

// captureGeminiRequest spins up a mock Gemini endpoint, runs a Stream with the
// given model ID + options, and returns the parsed request body.
func captureGeminiRequest(t *testing.T, modelID string, opts *stream.CallOptions) map[string]any {
	t.Helper()
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &received)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}` + "\n\n"))
	}))
	defer server.Close()

	m := createTestProvider(server.URL).Model(modelID)
	events, err := m.Stream(context.Background(), opts)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range events {
	}
	return received
}

func mustArray(t *testing.T, obj map[string]any, key string) []any {
	t.Helper()
	v, ok := obj[key].([]any)
	if !ok {
		t.Fatalf("%s is not an array: %v", key, obj[key])
	}
	return v
}
