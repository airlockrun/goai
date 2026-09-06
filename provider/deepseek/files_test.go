package deepseek

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
		if r.URL.Path != "/v1/files" || r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("X-Test") != "custom" {
			t.Errorf("request = %s, %v", r.URL, r.Header)
		}
		if err := r.ParseMultipartForm(1024); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		if r.FormValue("purpose") != "user_data" || r.FormValue("expires_after[seconds]") != "3600" {
			t.Errorf("form = %v", r.MultipartForm.Value)
		}
		io.WriteString(w, `{"id":"file-1","object":"file","bytes":4}`)
	}))
	defer server.Close()
	files := New(Options{APIKey: "key", BaseURL: server.URL + "/v1", Headers: map[string]string{"X-Test": "custom"}}).Files()
	if _, ok := files.(model.FileMetadataReader); ok {
		t.Fatal("upload-only capability exposes metadata")
	}
	result, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), MediaType: "image/png", ProviderOptions: map[string]any{"deepseek": map[string]any{"expiresAfter": 3600}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderReference["deepseek"] != "file-1" {
		t.Fatalf("result = %+v", result)
	}
}

func TestFilesValidation(t *testing.T) {
	files := New(Options{}).Files()
	for _, tc := range []struct {
		name, media, filename, data string
		options                     map[string]any
	}{
		{"pdf", "application/pdf", "a.pdf", "%PDF-1.7", nil},
		{"disguised pdf", "image/png", "a.png", "%PDF-1.7", nil},
		{"generic", "application/octet-stream", "a.bin", "unknown", nil},
		{"filename", "image/png", strings.Repeat("x", 513), "data", nil},
		{"expiry", "image/png", "a.png", "data", map[string]any{"deepseek": map[string]any{"expiresAfter": 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader(tc.data), MediaType: tc.media, Filename: tc.filename, ProviderOptions: tc.options}); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}
