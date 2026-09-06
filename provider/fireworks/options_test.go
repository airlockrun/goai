package fireworks

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/stream"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStructuredOutputAndTypedOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		format := body["response_format"].(map[string]any)
		if format["type"] != "json_schema" || format["json_schema"].(map[string]any)["strict"] != true {
			t.Errorf("response format = %v", format)
		}
		thinking := body["thinking"].(map[string]any)
		if thinking["budget_tokens"] != float64(2048) || body["reasoning_effort"] != "high" || body["prompt_cache_key"] != "cache" {
			t.Errorf("body = %v", body)
		}
		w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()
	events, err := New(Options{BaseURL: server.URL}).Model("m").Stream(context.Background(), &stream.CallOptions{ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: json.RawMessage(`{"type":"object"}`)}, ProviderOptions: map[string]any{"thinking": map[string]any{"type": "enabled", "budgetTokens": 2048}, "reasoningEffort": "xhigh", "promptCacheKey": "cache"}})
	if err != nil {
		t.Fatal(err)
	}
	for event := range events {
		if event.Type == stream.EventError {
			t.Fatal(event.Data)
		}
	}
}
