package vertex

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestVertexAudioRoutes(t *testing.T) {
	for _, tc := range []struct {
		id, path, response string
		speech             bool
	}{
		{"chirp-3-hd", "/v1/text:synthesize", `{"audioContent":"AQI="}`, true},
		{"gemini-2.5-flash-tts", "/publishers/google/models/gemini-2.5-flash-tts:generateContent", `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"AQI="}}]}}]}`, true},
		{"chirp_3", "/v2/projects/project/locations/us-central1/recognizers/_:recognize", `{"results":[{"languageCode":"en-US","alternatives":[{"transcript":"hello","words":[{"word":"hello","startOffset":"0s","endOffset":"1s"}]}]}],"metadata":{"totalBilledDuration":"1s"}}`, false},
		{"gemini-3.5-transcribe", "/publishers/google/models/gemini-3.5-transcribe:generateContent", `{"candidates":[{"content":{"parts":[{"audioTranscription":{"text":"hello","languageCode":"en","words":[{"word":"hello","startOffset":"0s","endOffset":"1s"}]}}]}}]}`, false},
	} {
		t.Run(tc.id, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.path || r.Header.Get("Authorization") != "Bearer token" {
					t.Errorf("request = %s %v", r.URL.Path, r.Header)
				}
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if tc.id == "gemini-3.5-transcribe" {
					if request["generationConfig"].(map[string]any)["audioTranscriptionConfig"].(map[string]any)["wordTimestamp"] != true {
						t.Errorf("request = %v", request)
					}
				}
				_, _ = w.Write([]byte(tc.response))
			}))
			defer server.Close()
			p := New(Options{BaseURL: server.URL, ProjectID: "project", AccessToken: "token"})
			if tc.speech {
				result, err := p.SpeechModel(tc.id).Generate(context.Background(), model.SpeechCallOptions{Text: "hello"})
				if err != nil {
					t.Fatal(err)
				}
				if len(result.Audio) == 0 || result.MimeType != "audio/wav" {
					t.Fatalf("result = %+v", result)
				}
			} else {
				result, err := p.TranscriptionModel(tc.id).Transcribe(context.Background(), model.TranscribeCallOptions{AudioReader: strings.NewReader("audio"), MimeType: "audio/wav", ProviderOptions: map[string]any{"wordTimestamp": true}})
				if err != nil {
					t.Fatal(err)
				}
				if result.Text != "hello" || len(result.Segments) != 1 || result.Segments[0].End != 1 {
					t.Fatalf("result = %+v", result)
				}
			}
		})
	}
}
