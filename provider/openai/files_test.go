package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
)

func TestFilesUpload(t *testing.T) {
	for _, tc := range []struct {
		name, purpose string
		options       map[string]any
	}{
		{"default", "assistants", nil},
		{"options", "user_data", map[string]any{"openai": FilesOptions{Purpose: "user_data", ExpiresAfter: func() *int64 { v := int64(3600); return &v }()}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/files" || r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("OpenAI-Organization") != "org" || r.Header.Get("OpenAI-Project") != "project" || r.Header.Get("X-Custom") != "call" {
					t.Errorf("request = %s, %v", r.URL, r.Header)
				}
				if err := r.ParseMultipartForm(1024); err != nil {
					t.Error(err)
					return
				}
				defer r.MultipartForm.RemoveAll()
				if r.FormValue("purpose") != tc.purpose {
					t.Errorf("purpose = %s", r.FormValue("purpose"))
				}
				if tc.options != nil && (r.FormValue("expires_after[anchor]") != "created_at" || r.FormValue("expires_after[seconds]") != "3600") {
					t.Error("missing expiration")
				}
				io.WriteString(w, `{"id":"file-1","filename":"test.txt","bytes":4}`)
			}))
			defer server.Close()
			files := New(provider.Options{APIKey: "key", BaseURL: server.URL + "/v1", Organization: "org", Project: "project", Headers: map[string]string{"X-Custom": "provider"}}).Files()
			if _, ok := files.(model.FileDownloader); !ok {
				t.Fatal("missing download capability")
			}
			result, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("test"), MediaType: "text/plain", Headers: map[string]string{"X-Custom": "call"}, ProviderOptions: tc.options})
			if err != nil {
				t.Fatal(err)
			}
			if result.ProviderReference["openai"] != "file-1" {
				t.Fatalf("result = %+v", result)
			}
		})
	}
}
