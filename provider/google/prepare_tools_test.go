package google

import (
	"github.com/airlockrun/goai/tool"
	"testing"
)

func TestFileSearchCapabilities(t *testing.T) {
	for _, tc := range []struct {
		id        string
		supported bool
	}{{"gemini-1.5-pro", false}, {"gemini-2.0-flash", false}, {"gemini-2.5-flash", true}, {"gemini-3.5-flash", true}, {"gemini-3.6-flash", true}, {"gemini-3.7-flash", true}, {"gemini-3.8-flash", true}, {"models/GEMINI-FUTURE-PRO", true}} {
		t.Run(tc.id, func(t *testing.T) {
			tools := prepareGeminiTools([]tool.Tool{FileSearch(FileSearchOptions{FileSearchStoreNames: []string{"fileSearchStores/test"}})}, tc.id)
			if (len(tools) == 1) != tc.supported {
				t.Fatalf("tools = %+v", tools)
			}
			if tc.supported && len(tools[0].FileSearch) == 0 {
				t.Fatal("missing fileSearch payload")
			}
		})
	}
}
