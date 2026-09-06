package elevenlabs

import (
	"context"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTranscription(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/speech-to-text" || r.Header.Get("xi-api-key") != "secret" {
			t.Errorf("request = %s %v", r.URL, r.Header)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		if r.FormValue("model_id") != "scribe_v2" || r.FormValue("diarize") != "false" || r.FormValue("language_code") != "en" {
			t.Errorf("form = %v", r.MultipartForm.Value)
		}
		w.Write([]byte(`{"text":"hello","language_code":"en","words":[{"text":"hello","start":0.1,"end":1.2}]}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL, APIKey: "secret"}).TranscriptionModel("scribe_v2").Transcribe(context.Background(), model.TranscribeCallOptions{Audio: []byte("audio"), Language: "en", ProviderOptions: map[string]any{"diarize": false}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || *result.Duration != 1.2 {
		t.Fatalf("result = %+v", result)
	}
}
