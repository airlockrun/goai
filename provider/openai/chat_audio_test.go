package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

func TestConvertUserContentAudio(t *testing.T) {
	content := message.Content{Parts: []message.Part{
		message.FilePart{Data: message.FileDataBytes{Data: "QUJD"}, MimeType: "audio/wav"},
	}}
	parts, ok := convertUserContent(content).([]chatContentPart)
	if !ok || len(parts) != 1 {
		t.Fatalf("expected 1 content part, got %#v", convertUserContent(content))
	}
	p := parts[0]
	if p.Type != "input_audio" || p.InputAudio == nil {
		t.Fatalf("expected input_audio part, got %+v", p)
	}
	if p.InputAudio.Data != "QUJD" || p.InputAudio.Format != "wav" {
		t.Errorf("input_audio = %+v, want data=QUJD format=wav", p.InputAudio)
	}
}

// audioServer returns a streaming chat-completions stub (audio output requires
// stream:true) that records the decoded request and replies with SSE chunks:
// the audio bytes arrive split across two deltas to exercise accumulation.
func audioServer(t *testing.T, gotReq *chatRequest, audioB64, content string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, gotReq); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE := func(v any) { b, _ := json.Marshal(v); io.WriteString(w, "data: "+string(b)+"\n\n") }
		chunk := func(delta map[string]any) map[string]any {
			return map[string]any{"choices": []map[string]any{{"delta": delta}}}
		}
		if content != "" {
			writeSSE(chunk(map[string]any{"content": content}))
		}
		if audioB64 != "" {
			mid := len(audioB64) / 2
			writeSSE(chunk(map[string]any{"audio": map[string]any{"data": audioB64[:mid], "transcript": "spoken "}}))
			writeSSE(chunk(map[string]any{"audio": map[string]any{"data": audioB64[mid:], "transcript": "transcript"}}))
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}))
}

func TestChatSpeechModelGenerate(t *testing.T) {
	want := []byte("fake-audio-bytes")
	var gotReq chatRequest
	srv := audioServer(t, &gotReq, base64.StdEncoding.EncodeToString(want), "")
	defer srv.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: srv.URL})
	res, err := p.ChatSpeechModel("gpt-4o-mini-audio-preview").Generate(context.Background(), model.SpeechCallOptions{
		Text:         "Hello",
		Voice:        "verse",
		OutputFormat: "pcm16",
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	// Request shape: modalities must include both text and audio; audio config
	// set; streaming required (audio output rejects stream:false).
	if len(gotReq.Modalities) != 2 || gotReq.Modalities[0] != "text" || gotReq.Modalities[1] != "audio" {
		t.Errorf("modalities = %v, want [text audio]", gotReq.Modalities)
	}
	if gotReq.Audio == nil || gotReq.Audio.Voice != "verse" || gotReq.Audio.Format != "pcm16" {
		t.Errorf("audio config = %+v, want voice=verse format=pcm16", gotReq.Audio)
	}
	if !gotReq.Stream {
		t.Error("chat-audio adapter must stream (audio output requires stream:true)")
	}
	if !bytes.Contains(res.Audio, want) {
		t.Errorf("wav output should contain the pcm payload %q", want)
	}
	if res.MimeType != "audio/wav" {
		t.Errorf("mime = %q, want audio/wav", res.MimeType)
	}
}

func TestChatSpeechModelFormatFallback(t *testing.T) {
	var gotReq chatRequest
	srv := audioServer(t, &gotReq, base64.StdEncoding.EncodeToString([]byte("x")), "")
	defer srv.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: srv.URL})
	res, err := p.ChatSpeechModel("gpt-audio").Generate(context.Background(), model.SpeechCallOptions{
		Text:         "Hi",
		OutputFormat: "wav", // not allowed when streaming → forced to pcm16 + warning
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotReq.Audio.Format != "pcm16" {
		t.Errorf("format = %q, want pcm16", gotReq.Audio.Format)
	}
	if len(res.Warnings) == 0 {
		t.Error("expected an unsupported-format warning")
	}
}

func TestChatTranscriptionModel(t *testing.T) {
	var gotReq chatRequest
	srv := audioServer(t, &gotReq, "", "the transcribed words")
	defer srv.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: srv.URL})
	res, err := p.ChatTranscriptionModel("gpt-4o-mini-audio-preview").Transcribe(context.Background(), model.TranscribeCallOptions{
		Audio:    []byte("raw-audio"),
		MimeType: "audio/wav",
		Prompt:   "Echo the user message verbatim.", // caller-supplied instruction, not baked into goai
	})
	if err != nil {
		t.Fatalf("transcribe: %v", err)
	}
	// STT requests text-only output; audio rides as input_audio in the user turn.
	if len(gotReq.Modalities) != 1 || gotReq.Modalities[0] != "text" {
		t.Errorf("modalities = %v, want [text]", gotReq.Modalities)
	}
	// opts.Prompt becomes a system message; goai injects none of its own.
	if len(gotReq.Messages) != 2 || gotReq.Messages[0].Role != "system" {
		t.Fatalf("expected [system, user] messages, got %d", len(gotReq.Messages))
	}
	if res.Text != "the transcribed words" {
		t.Errorf("text = %q", res.Text)
	}
}

// TestChatTranscriptionNoPrompt: with no caller instruction, only the audio turn
// is sent — goai supplies no system message of its own.
func TestChatTranscriptionNoPrompt(t *testing.T) {
	var gotReq chatRequest
	srv := audioServer(t, &gotReq, "", "reply")
	defer srv.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: srv.URL})
	_, err := p.ChatTranscriptionModel("gpt-audio").Transcribe(context.Background(), model.TranscribeCallOptions{
		Audio: []byte("raw-audio"), MimeType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("transcribe: %v", err)
	}
	if len(gotReq.Messages) != 1 || gotReq.Messages[0].Role != "user" {
		t.Fatalf("expected only a user message, got %d", len(gotReq.Messages))
	}
}
