package xai

import (
	"encoding/json"
	"testing"
)

func TestUsageRawAndClamping(t *testing.T) {
	var usage responsesUsage
	if err := json.Unmarshal([]byte(`{"input_tokens":10,"output_tokens":2,"output_tokens_details":{"reasoning_tokens":5,"custom":7},"cost":0.1}`), &usage); err != nil {
		t.Fatal(err)
	}
	got := convertXaiResponsesUsage(&usage)
	if *got.OutputTokens.Text != 0 || *got.OutputTokens.Reasoning != 5 || got.Raw["cost"] != 0.1 || got.Raw["output_tokens_details"].(map[string]any)["custom"] != float64(7) {
		t.Fatalf("usage = %+v", got)
	}
}
