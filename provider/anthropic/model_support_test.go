package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/stream"
)

func TestModelSupport(t *testing.T) {
	for _, tc := range []struct {
		id             string
		limit          int
		known, rejects bool
	}{
		{"claude-sonnet-5", 128000, true, true}, {"claude-opus-5", 128000, true, true}, {"claude-fable-5-1", 128000, true, true},
		{"us.anthropic.claude-opus-4-8-v1:0", 128000, true, true}, {"claude-sonnet-4-6@20260101", 128000, true, false},
		{"claude-haiku-4-5", 64000, true, false}, {"claude-opus-4-1", 32000, true, false}, {"claude-3-haiku", 4096, true, false},
		{"claude-future-9", 128000, false, true}, {"claude-3-7-sonnet", 4096, false, false}, {"future-model", 4096, false, false},
	} {
		t.Run(tc.id, func(t *testing.T) {
			limit, known, rejects := modelSupport(tc.id)
			if limit != tc.limit || known != tc.known || rejects != tc.rejects {
				t.Fatalf("%d %v %v", limit, known, rejects)
			}
			temp, topP, topK := 0.5, 0.9, 20
			raw, _, warnings, err := BuildRequestBody(Config{}, tc.id, &stream.CallOptions{Temperature: &temp, TopP: &topP, TopK: &topK})
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatal(err)
			}
			if body["max_tokens"] != float64(tc.limit) {
				t.Fatalf("%s", raw)
			}
			if tc.rejects && (body["temperature"] != nil || body["top_p"] != nil || body["top_k"] != nil) {
				t.Fatalf("sampling: %s", raw)
			}
			if !tc.known && len(warnings) == 0 {
				t.Fatal("missing unknown model warning")
			}
		})
	}
}

func TestModelOutputLimits(t *testing.T) {
	for _, tc := range []struct {
		id   string
		want int
	}{{"claude-sonnet-5", 128000}, {"unknown-model", 200000}} {
		t.Run(tc.id, func(t *testing.T) {
			max := 200000
			raw, _, _, err := BuildRequestBody(Config{}, tc.id, &stream.CallOptions{MaxOutputTokens: &max})
			if err != nil {
				t.Fatal(err)
			}
			var b map[string]any
			_ = json.Unmarshal(raw, &b)
			if b["max_tokens"] != float64(tc.want) {
				t.Fatalf("%s", raw)
			}
		})
	}
}

func TestModernStructuredOutputDefaults(t *testing.T) {
	for _, id := range []string{"claude-sonnet-5", "claude-opus-5", "claude-fable-5-1", "claude-future-9"} {
		t.Run(id, func(t *testing.T) {
			raw, _, _, err := BuildRequestBody(Config{}, id, &stream.CallOptions{ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)}})
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			_ = json.Unmarshal(raw, &body)
			if body["output_config"] == nil || body["tools"] != nil {
				t.Fatalf("%s", raw)
			}
		})
	}
}
