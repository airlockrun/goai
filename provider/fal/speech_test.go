package fal

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSpeechDownload(t *testing.T) {
	var base string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/audio" {
			if r.Header.Get("Authorization") != "" {
				t.Error("credential leaked to audio download")
			}
			w.Header().Set("Content-Type", "audio/wav")
			w.Write([]byte("audio"))
			return
		}
		if r.URL.Path != "/fal-ai/tts" || r.Header.Get("Authorization") != "Key secret" {
			t.Errorf("request = %s %v", r.URL, r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["text"] != "hello" || body["output_format"] != "url" {
			t.Errorf("body = %v", body)
		}
		json.NewEncoder(w).Encode(map[string]any{"audio": map[string]string{"url": base + "/audio"}, "duration_ms": 1500})
	}))
	defer server.Close()
	base = server.URL
	result, err := New(Options{BaseURL: server.URL, APIKey: "secret"}).SpeechModel("fal-ai/tts").Generate(context.Background(), model.SpeechCallOptions{Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if string(result.Audio) != "audio" || result.MimeType != "audio/wav" || *result.Duration != 1.5 {
		t.Fatalf("result = %+v", result)
	}
}
