package xai

import (
	"context"
	"github.com/airlockrun/goai/model"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTranscriptionFileLast(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stt" {
			t.Errorf("path = %s", r.URL.Path)
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Error(err)
			return
		}
		last := ""
		terms := 0
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Error(err)
				return
			}
			last = part.FormName()
			if last == "keyterm" {
				terms++
			}
			data, _ := io.ReadAll(part)
			if last == "file" && string(data) != "audio" {
				t.Errorf("audio = %q", data)
			}
		}
		if last != "file" || terms != 2 {
			t.Errorf("last=%s keyterms=%d", last, terms)
		}
		w.Write([]byte(`{"text":"hello","language":"en","duration":2,"words":[{"text":"hello","start":0.1,"end":1.5}]}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL}).TranscriptionModel("").Transcribe(context.Background(), model.TranscribeCallOptions{Audio: []byte("audio"), MimeType: "audio/wav", ProviderOptions: map[string]any{"keyterm": []string{"one", "two"}, "diarize": false}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" || len(result.Segments) != 1 || result.Segments[0].End != 1.5 {
		t.Fatalf("result = %+v", result)
	}
}
