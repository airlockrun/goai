package goai

import (
	"testing"

	"github.com/airlockrun/goai/message"
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

func TestBuildCallOptions_PrependsInstructions(t *testing.T) {
	t.Run("instructions become the leading system message", func(t *testing.T) {
		input := stream.Input{
			Instructions: "be terse",
			Messages:     []message.Message{message.NewUserMessage("hi")},
		}
		opts := buildCallOptions(&input)
		if len(opts.Messages) != 2 {
			t.Fatalf("got %d messages, want 2", len(opts.Messages))
		}
		if opts.Messages[0].Role != message.RoleSystem || opts.Messages[0].Content.Text != "be terse" {
			t.Errorf("leading message = %+v, want system 'be terse'", opts.Messages[0])
		}
		if opts.Messages[1].Role != message.RoleUser {
			t.Errorf("second message role = %q, want user", opts.Messages[1].Role)
		}
	})

	t.Run("empty instructions leaves messages untouched", func(t *testing.T) {
		input := stream.Input{Messages: []message.Message{message.NewUserMessage("hi")}}
		opts := buildCallOptions(&input)
		if len(opts.Messages) != 1 {
			t.Fatalf("got %d messages, want 1", len(opts.Messages))
		}
	})
}
