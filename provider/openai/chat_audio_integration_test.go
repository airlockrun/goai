package openai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

// chatAudioRig returns a provider + chat-audio model id, preferring a direct
// OpenAI key and falling back to OpenRouter's OpenAI-compatible endpoint (the
// key we have on hand). Skips when neither key is available. The audio adapters
// are constructed directly so the OpenRouter-prefixed id still routes here.
func chatAudioRig(t *testing.T) (*Provider, string) {
	t.Helper()
	if k := os.Getenv("OPENAI_API_KEY"); k != "" {
		return New(provider.Options{APIKey: k}), "gpt-audio-mini"
	}
	if k := readKey("OPENROUTER_API_KEY"); k != "" {
		return New(provider.Options{APIKey: k, BaseURL: "https://openrouter.ai/api/v1"}),
			"openai/gpt-audio"
	}
	t.Skip("no OPENAI_API_KEY or OPENROUTER_API_KEY (env or .env)")
	return nil, ""
}

func readKey(name string) string {
	if k := os.Getenv(name); k != "" {
		return k
	}
	for _, p := range []string{".env", "../../.env"} {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			if v, ok := strings.CutPrefix(strings.TrimSpace(line), name+"="); ok {
				return strings.Trim(strings.TrimSpace(v), `"'`)
			}
		}
	}
	return ""
}

// TestIntegration_ChatAudioSpeech drives the chat-backed SpeechModel adapter:
// text in, spoken audio out via /chat/completions (modalities + audio).
func TestIntegration_ChatAudioSpeech(t *testing.T) {
	p, id := chatAudioRig(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	m := p.ChatSpeechModel(id)
	res, err := m.Generate(ctx, model.SpeechCallOptions{
		Text:  "Hello from the chat-audio integration test.",
		Voice: "alloy",
	})
	if err != nil {
		t.Fatalf("chat speech generate: %v", err)
	}
	if len(res.Audio) == 0 {
		t.Fatal("expected audio bytes, got none")
	}
	t.Logf("chat-audio TTS: %d bytes, %s", len(res.Audio), res.MimeType)
}

// TestIntegration_ChatAudioTranscribe round-trips: synthesize speech, then feed
// it back through the chat-backed TranscriptionModel. The transcribe instruction
// is supplied by the caller via opts.Prompt — goai bakes none in. (gpt-audio is
// conversational, so even with the instruction it may editorialize; the test
// only asserts non-empty text.)
func TestIntegration_ChatAudioTranscribe(t *testing.T) {
	p, id := chatAudioRig(t)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const phrase = "The meeting is scheduled for three fifteen on Thursday afternoon."
	out, err := p.ChatSpeechModel(id).Generate(ctx, model.SpeechCallOptions{Text: phrase})
	if err != nil {
		t.Fatalf("tts: %v", err)
	}

	tr, err := p.ChatTranscriptionModel(id).Transcribe(ctx, model.TranscribeCallOptions{
		Audio:    out.Audio, // already WAV (the speech adapter wraps pcm16)
		MimeType: "audio/wav",
		Prompt:   "Echo back the user message verbatim as text. Output only what is said.",
	})
	if err != nil {
		t.Fatalf("stt: %v", err)
	}
	if strings.TrimSpace(tr.Text) == "" {
		t.Fatal("expected transcription text, got empty")
	}
	t.Logf("chat-audio STT: %q", tr.Text)
}
