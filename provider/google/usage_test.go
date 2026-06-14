package google

import "testing"

func deref(p *int) int {
	if p == nil {
		return -1
	}
	return *p
}

func TestGeminiUsageMetadata_ToUsage(t *testing.T) {
	tests := []struct {
		name string
		meta geminiUsageMetadata
		// -1 means the pointer should be nil (unset)
		wantInTotal, wantNoCache, wantCacheRead int
		wantOutTotal, wantText, wantReasoning   int
	}{
		{
			name:          "plain",
			meta:          geminiUsageMetadata{PromptTokenCount: 100, CandidatesTokenCount: 40},
			wantInTotal:   100,
			wantNoCache:   100,
			wantCacheRead: -1,
			wantOutTotal:  40,
			wantText:      40,
			wantReasoning: -1,
		},
		{
			name:          "cached content",
			meta:          geminiUsageMetadata{PromptTokenCount: 100, CandidatesTokenCount: 40, CachedContentTokenCount: 70},
			wantInTotal:   100,
			wantNoCache:   30,
			wantCacheRead: 70,
			wantOutTotal:  40,
			wantText:      40,
			wantReasoning: -1,
		},
		{
			name:          "thoughts add to output total",
			meta:          geminiUsageMetadata{PromptTokenCount: 100, CandidatesTokenCount: 40, CachedContentTokenCount: 70, ThoughtsTokenCount: 25},
			wantInTotal:   100,
			wantNoCache:   30,
			wantCacheRead: 70,
			wantOutTotal:  65, // 40 candidates + 25 thoughts
			wantText:      40,
			wantReasoning: 25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := tt.meta.toUsage()
			if got := deref(u.InputTokens.Total); got != tt.wantInTotal {
				t.Errorf("InputTokens.Total = %d, want %d", got, tt.wantInTotal)
			}
			if got := deref(u.InputTokens.NoCache); got != tt.wantNoCache {
				t.Errorf("InputTokens.NoCache = %d, want %d", got, tt.wantNoCache)
			}
			if got := deref(u.InputTokens.CacheRead); got != tt.wantCacheRead {
				t.Errorf("InputTokens.CacheRead = %d, want %d", got, tt.wantCacheRead)
			}
			if got := deref(u.OutputTokens.Total); got != tt.wantOutTotal {
				t.Errorf("OutputTokens.Total = %d, want %d", got, tt.wantOutTotal)
			}
			if got := deref(u.OutputTokens.Text); got != tt.wantText {
				t.Errorf("OutputTokens.Text = %d, want %d", got, tt.wantText)
			}
			if got := deref(u.OutputTokens.Reasoning); got != tt.wantReasoning {
				t.Errorf("OutputTokens.Reasoning = %d, want %d", got, tt.wantReasoning)
			}
		})
	}
}
