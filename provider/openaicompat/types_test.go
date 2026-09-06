package openaicompat

import (
	"encoding/json"
	"github.com/airlockrun/goai/message"
	"testing"
)

func TestVideoContent(t *testing.T) {
	for _, tc := range []struct {
		name string
		data message.FileData
		want string
	}{
		{"url", message.FileDataURL{URL: "https://example.com/movie.mp4"}, "https://example.com/movie.mp4"},
		{"bytes", message.FileDataBytes{Data: "YQ=="}, "data:video/mp4;base64,YQ=="},
	} {
		t.Run(tc.name, func(t *testing.T) {
			content := convertUserContent(message.Content{Parts: []message.Part{message.FilePart{MimeType: "video/mp4", Data: tc.data}}})
			data, err := json.Marshal(content)
			if err != nil {
				t.Fatal(err)
			}
			var parts []struct {
				Type  string `json:"type"`
				Video struct {
					URL string `json:"url"`
				} `json:"video_url"`
			}
			if err := json.Unmarshal(data, &parts); err != nil {
				t.Fatal(err)
			}
			if len(parts) != 1 || parts[0].Type != "video_url" || parts[0].Video.URL != tc.want {
				t.Fatalf("content = %s", data)
			}
		})
	}
}
