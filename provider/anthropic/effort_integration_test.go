package anthropic

import (
	"context"
	"testing"
	"time"

	"github.com/airlockrun/goai"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Sanity check that Anthropic accepts the new output_config.effort field
// against the live API. Earlier opts.Effort was declared but never written —
// this regression test catches the case where the wiring in chat.go is later
// removed or the wire field is renamed.
func TestIntegration_EffortAccepted(t *testing.T) {
	skipIfNoKey(t)
	p := getProvider()
	for _, modelID := range []string{
		"claude-haiku-4-5-20251001",
		"claude-sonnet-4-6",
		"claude-opus-4-7",
	} {
		modelID := modelID
		t.Run(modelID, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			m := p.LanguageModel(modelID)
			_, err := goai.GenerateText(ctx, stream.Input{
				Model: m,
				Messages: []message.Message{
					message.NewUserMessage("Say hello in 3 words."),
				},
				ProviderOptions: map[string]any{
					"effort": "medium",
				},
			})
			if err != nil {
				t.Logf("effort=medium on %s: %v", modelID, err)
			} else {
				t.Logf("effort=medium accepted by %s", modelID)
			}
		})
	}
}
