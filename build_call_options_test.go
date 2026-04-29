package goai

import (
	"testing"

	"github.com/airlockrun/goai/stream"
)

// buildCallOptions plumbs the new v4 fields (IncludeRawChunks, Reasoning)
// from Input to CallOptions alongside ProviderOptions / Headers / etc.
// Mirrors ai-sdk v4 LanguageModelV4CallOptions.

func TestBuildCallOptions_PassesIncludeRawChunks(t *testing.T) {
	for _, want := range []bool{false, true} {
		input := stream.Input{IncludeRawChunks: want}
		opts := buildCallOptions(&input)
		if opts.IncludeRawChunks != want {
			t.Errorf("IncludeRawChunks: got %v, want %v", opts.IncludeRawChunks, want)
		}
	}
}

func TestBuildCallOptions_PassesReasoning(t *testing.T) {
	for _, want := range []stream.ReasoningEffort{
		"", // provider-default sentinel (empty)
		stream.ReasoningEffortNone,
		stream.ReasoningEffortMinimal,
		stream.ReasoningEffortLow,
		stream.ReasoningEffortMedium,
		stream.ReasoningEffortHigh,
		stream.ReasoningEffortXHigh,
	} {
		input := stream.Input{Reasoning: want}
		opts := buildCallOptions(&input)
		if opts.Reasoning != want {
			t.Errorf("Reasoning: got %q, want %q", opts.Reasoning, want)
		}
	}
}
