package google

import "testing"

func TestThinkingConfiguration(t *testing.T) {
	for _, tc := range []struct{ id, want string }{{"gemini-3.5-flash", "minimal"}, {"gemini-3.6-flash", "minimal"}, {"gemini-3.7-flash", "low"}, {"gemini-3.8-flash", "low"}, {"models/GEMINI-3.8-FLASH-LITE", "minimal"}, {"gemini-4.0-flash", "low"}, {"gemini-flash-latest", "low"}, {"gemini-future-pro", "minimal"}} {
		t.Run(tc.id, func(t *testing.T) {
			got, err := ThinkingConfiguration(tc.id, "none", nil)
			if err != nil || got["thinkingLevel"] != tc.want {
				t.Fatalf("got %v, %v", got, err)
			}
		})
	}
	t.Run("explicit zero", func(t *testing.T) {
		got, err := ThinkingConfiguration("gemini-3.8-flash", "high", map[string]any{"thinkingBudget": 0})
		if err != nil || got["thinkingBudget"] != float64(0) || got["thinkingLevel"] != nil {
			t.Fatalf("got %v, %v", got, err)
		}
	})
}
