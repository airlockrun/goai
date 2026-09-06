package xai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

func TestFilesUpload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("X-Test") != "custom" {
			t.Errorf("headers = %v", r.Header)
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Error(err)
			return
		}
		for _, want := range [][2]string{{"expires_after", "3600"}, {"team_id", "team"}, {"file", "data"}} {
			part, err := reader.NextPart()
			if err != nil {
				t.Error(err)
				return
			}
			data, err := io.ReadAll(part)
			if err != nil || part.FormName() != want[0] || string(data) != want[1] {
				t.Errorf("part = %s, %s, %v", part.FormName(), data, err)
			}
		}
		io.WriteString(w, `{"id":"file-1","bytes":4}`)
	}))
	defer server.Close()
	files := New(Options{APIKey: "key", BaseURL: server.URL, Headers: map[string]string{"X-Test": "custom"}}).Files()
	if _, ok := files.(model.FileDeleter); !ok {
		t.Fatal("missing deletion capability")
	}
	result, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), MediaType: "text/plain", ProviderOptions: map[string]any{"xai": map[string]any{"expiresAfter": 3600, "teamId": "team"}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderReference["xai"] != "file-1" {
		t.Fatalf("result = %+v", result)
	}
	for _, value := range []any{1, 2592001, 3600.5, "3600"} {
		if _, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), MediaType: "text/plain", ProviderOptions: map[string]any{"xai": map[string]any{"expiresAfter": value}}}); err == nil {
			t.Errorf("accepted expiration %v", value)
		}
	}
}
