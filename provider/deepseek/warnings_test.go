package deepseek

import (
	"testing"

	"github.com/airlockrun/goai/stream"
)

func intPtr(v int) *int { return &v }

func TestDeepSeekCallWarner_TopK(t *testing.T) {
	warnings := deepseekCallWarner(&stream.CallOptions{TopK: intPtr(5)})
	found := false
	for _, w := range warnings {
		if w.Feature == "topK" && w.Type == stream.WarningUnsupported {
			found = true
		}
	}
	if !found {
		t.Errorf("expected topK warning, got %+v", warnings)
	}
}
