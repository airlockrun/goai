package mistral

import (
	"context"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTranscription(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		if r.FormValue("model") != "voxtral" || r.FormValue("timestamp_granularities") != "segment" || len(r.MultipartForm.Value["context_bias"]) != 2 {
			t.Errorf("form = %v", r.MultipartForm.Value)
		}
		w.Write([]byte(`{"model":"voxtral","text":"hello","language":"en","segments":[{"text":"hello","start":0,"end":2}],"usage":{"prompt_audio_seconds":3}}`))
	}))
	defer server.Close()
	m := New(Options{BaseURL: server.URL}).TranscriptionModel("voxtral")
	result, err := m.Transcribe(context.Background(), model.TranscribeCallOptions{Audio: []byte("audio"), ProviderOptions: map[string]any{"timestampGranularities": []string{"segment"}, "contextBias": []string{"a", "b"}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || *result.Duration != 3 {
		t.Fatalf("result = %+v", result)
	}
	if _, err := m.Transcribe(context.Background(), model.TranscribeCallOptions{Language: "en", ProviderOptions: map[string]any{"timestampGranularities": []string{"segment"}}}); err == nil {
		t.Fatal("expected incompatible option error")
	}
}
