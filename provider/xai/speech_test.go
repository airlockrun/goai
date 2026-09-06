package xai

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSpeechTimestamps(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tts" || r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("request = %s %v", r.URL, r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["voice_id"] != "eve" || body["language"] != "auto" || body["text_normalization"] != false || body["with_timestamps"] != true {
			t.Errorf("body = %v", body)
		}
		if _, ok := body["model"]; ok {
			t.Error("xAI tts does not accept model")
		}
		w.Write([]byte(`{"audio":"YXVkaW8=","content_type":"audio/mpeg","duration":1.5}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL, APIKey: "key"}).SpeechModel("").Generate(context.Background(), model.SpeechCallOptions{Text: "hello", ProviderOptions: map[string]any{"withTimestamps": true, "textNormalization": false}})
	if err != nil {
		t.Fatal(err)
	}
	if string(result.Audio) != "audio" || result.Duration == nil || *result.Duration != 1.5 {
		t.Fatalf("result = %+v", result)
	}
}
