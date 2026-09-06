package mistral

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSpeechReferenceAudio(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/speech" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["ref_audio"] != "reference" || body["voice_id"] != nil || body["stream"] != false || body["response_format"] != "wav" {
			t.Errorf("body = %v", body)
		}
		w.Write([]byte(`{"audio_data":"YXVkaW8="}`))
	}))
	defer server.Close()
	speed := 1.2
	result, err := New(Options{BaseURL: server.URL}).SpeechModel("voxtral-mini-tts").Generate(context.Background(), model.SpeechCallOptions{Text: "hello", Voice: "voice", OutputFormat: "wav", Speed: &speed, ProviderOptions: map[string]any{"refAudio": "reference"}})
	if err != nil {
		t.Fatal(err)
	}
	if string(result.Audio) != "audio" || result.MimeType != "audio/wav" || len(result.Warnings) != 1 {
		t.Fatalf("result = %+v", result)
	}
}
