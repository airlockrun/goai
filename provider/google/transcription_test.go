package google

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGoogleTranscription(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/interactions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		config := request["generation_config"].(map[string]any)["transcription_config"].(map[string]any)
		if config["mode"].(map[string]any)["type"] != "smart" {
			t.Errorf("config = %v", config)
		}
		_, _ = w.Write([]byte(`{"steps":[{"content":[{"type":"text","text":"hello","annotations":[{"type":"word_info","text":"hello","start_offset":"0s","end_offset":"1.2s"}]}]}]}`))
	}))
	defer server.Close()
	p := New(Options{BaseURL: server.URL})
	result, err := p.TranscriptionModel("gemini-3.5-transcribe").Transcribe(context.Background(), model.TranscribeCallOptions{Audio: []byte("audio"), MimeType: "audio/wav", ProviderOptions: map[string]any{"mode": "SMART", "wordTimestamp": true}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || len(result.Segments) != 1 || result.Segments[0].End != 1.2 {
		t.Fatalf("result = %+v", result)
	}
	if _, err := p.TranscriptionModel("gemini-3.5-transcribe-live").Transcribe(context.Background(), model.TranscribeCallOptions{}); err == nil {
		t.Fatal("live unary must fail")
	}
}
