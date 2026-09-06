package cerebras

import "testing"

func TestTypedOptions(t *testing.T) {
	fields, _, err := cerebrasRequestModifier(map[string]any{"parallelToolCalls": false, "topLogprobs": 0, "reasoningEffort": "high", "serviceTier": "priority", "promptCacheKey": "cache", "reasoningFormat": "parsed"})
	if err != nil {
		t.Fatal(err)
	}
	if fields["parallel_tool_calls"] != false || fields["top_logprobs"] != 0 || fields["prompt_cache_key"] != "cache" || fields["reasoning_effort"] != "high" {
		t.Fatalf("fields = %v", fields)
	}
	for _, tc := range []struct {
		name string
		opts map[string]any
	}{
		{"logprobs", map[string]any{"topLogprobs": 21}},
		{"bias", map[string]any{"logitBias": map[string]any{"1": 101}}},
		{"tier", map[string]any{"serviceTier": "invalid"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := cerebrasRequestModifier(tc.opts); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}
