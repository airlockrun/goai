package openaicompat

import (
	"testing"
)

func deref(p *int) int {
	if p == nil {
		return -1 // sentinel: distinguishes "unset" from a real 0
	}
	return *p
}

func TestUsageFromChat(t *testing.T) {
	tests := []struct {
		name                                  string
		usage                                 chatUsage
		wantTotal, wantNoCache, wantCacheRead int
		wantOutTotal, wantText, wantReasoning int
	}{
		{
			name:          "plain totals only",
			usage:         chatUsage{PromptTokens: 100, CompletionTokens: 50},
			wantTotal:     100,
			wantNoCache:   100,
			wantCacheRead: -1,
			wantOutTotal:  50,
			wantText:      50,
			wantReasoning: -1,
		},
		{
			name:          "cache via prompt_tokens_details (openai/deepseek)",
			usage:         chatUsage{PromptTokens: 339, CompletionTokens: 83, PromptTokensDetails: &promptTokensDetails{CachedTokens: 320}},
			wantTotal:     339,
			wantNoCache:   19,
			wantCacheRead: 320,
			wantOutTotal:  83,
			wantText:      83,
			wantReasoning: -1,
		},
		{
			name:          "cache via num_cached_tokens (mistral)",
			usage:         chatUsage{PromptTokens: 200, CompletionTokens: 40, NumCachedTokens: 120},
			wantTotal:     200,
			wantNoCache:   80,
			wantCacheRead: 120,
			wantOutTotal:  40,
			wantText:      40,
			wantReasoning: -1,
		},
		{
			name:          "cache via prompt_cache_hit_tokens (deepseek-native)",
			usage:         chatUsage{PromptTokens: 339, CompletionTokens: 83, PromptCacheHitTokens: 320},
			wantTotal:     339,
			wantNoCache:   19,
			wantCacheRead: 320,
			wantOutTotal:  83,
			wantText:      83,
			wantReasoning: -1,
		},
		{
			name:          "reasoning split (deepseek-reasoner)",
			usage:         chatUsage{PromptTokens: 339, CompletionTokens: 83, PromptTokensDetails: &promptTokensDetails{CachedTokens: 320}, CompletionTokensDetails: &completionTokensDetails{ReasoningTokens: 39}},
			wantTotal:     339,
			wantNoCache:   19,
			wantCacheRead: 320,
			wantOutTotal:  83,
			wantText:      44, // 83 - 39
			wantReasoning: 39,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := usageFromChat(tt.usage)
			if got := deref(u.InputTokens.Total); got != tt.wantTotal {
				t.Errorf("InputTokens.Total = %d, want %d", got, tt.wantTotal)
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
