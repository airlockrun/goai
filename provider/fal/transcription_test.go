package fal

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTranscriptionQueue(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Key secret" {
			t.Error("missing auth")
		}
		switch r.URL.Path {
		case "/fal-ai/whisper":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body["audio_url"] != "data:audio/wav;base64,YXVkaW8=" || body["diarize"] != false || body["chunk_level"] != "word" {
				t.Errorf("body = %v", body)
			}
			w.Write([]byte(`{"request_id":"job-1"}`))
		case "/fal-ai/whisper/requests/job-1":
			if r.Method != "GET" {
				t.Error("poll must use GET")
			}
			w.Write([]byte(`{"text":"hello","inferred_languages":["en"],"chunks":[{"text":"hello","timestamp":[0.1,1.2]}]}`))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL, APIKey: "secret"}).TranscriptionModel("whisper").Transcribe(context.Background(), model.TranscribeCallOptions{Audio: []byte("audio"), MimeType: "audio/wav", ProviderOptions: map[string]any{"diarize": false}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || result.Language != "en" || *result.Duration != 1.2 {
		t.Fatalf("result = %+v", result)
	}
}
