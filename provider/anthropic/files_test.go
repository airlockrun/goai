package anthropic

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
		if r.Header.Get("X-Api-Key") != "key" || r.Header.Get("Anthropic-Version") != apiVersion || r.Header.Get("Anthropic-Beta") != "files-api-2025-04-14" {
			t.Errorf("headers = %v", r.Header)
		}
		if err := r.ParseMultipartForm(1024); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		if r.FormValue("purpose") != "" {
			t.Error("unexpected purpose")
		}
		io.WriteString(w, `{"id":"file-1","type":"file","filename":"test.txt","mime_type":"text/plain","size_bytes":4,"created_at":"2026-01-01T00:00:00Z","downloadable":false}`)
	}))
	defer server.Close()
	files := New(Options{APIKey: "key", BaseURL: server.URL}).Files()
	if _, ok := files.(model.FileDownloader); ok {
		t.Fatal("upload-only capability exposes download")
	}
	result, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), MediaType: "text/plain"})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderReference["anthropic"] != "file-1" || result.CreatedAt == nil || *result.ByteSize != 4 || result.ProviderMetadata["anthropic"].(map[string]any)["downloadable"] != false {
		t.Fatalf("result = %+v", result)
	}
}
