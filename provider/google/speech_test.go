package google

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGoogleSpeechGenerate(t *testing.T) {
	for _, format := range []string{"wav", "pcm"} {
		t.Run(format, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/models/gemini-2.5-flash-preview-tts:generateContent" {
					t.Errorf("path = %s", r.URL.Path)
				}
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				config := request["generationConfig"].(map[string]any)
				if config["responseModalities"].([]any)[0] != "AUDIO" {
					t.Errorf("config = %v", config)
				}
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;rate=16000","data":"AQIDBA=="}}]}}]}`))
			}))
			defer server.Close()
			result, err := New(Options{BaseURL: server.URL}).SpeechModel("gemini-2.5-flash-preview-tts").Generate(context.Background(), model.SpeechCallOptions{Text: "hello", OutputFormat: format})
			if err != nil {
				t.Fatal(err)
			}
			if format == "wav" {
				if len(result.Audio) != 48 || string(result.Audio[:4]) != "RIFF" || binary.LittleEndian.Uint32(result.Audio[24:]) != 16000 || result.MimeType != "audio/wav" {
					t.Fatalf("result = %+v", result)
				}
			} else if len(result.Audio) != 4 {
				t.Fatalf("audio = %v", result.Audio)
			}
		})
	}
}
