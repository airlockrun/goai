package google

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

func TestFilesUpload(t *testing.T) {
	for _, state := range []string{"ACTIVE", "FAILED", "PROCESSING"} {
		t.Run(state, func(t *testing.T) {
			var server *httptest.Server
			server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/upload/v1beta/files":
					if r.Header.Get("X-Goog-Api-Key") != "key" || r.Header.Get("X-Goog-Upload-Protocol") != "resumable" || r.Header.Get("X-Goog-Upload-Header-Content-Length") != "4" {
						t.Errorf("headers = %v", r.Header)
					}
					body, _ := io.ReadAll(r.Body)
					if !strings.Contains(string(body), `"display_name":"test"`) {
						t.Errorf("body = %s", body)
					}
					w.Header().Set("X-Goog-Upload-Url", server.URL+"/session")
				case "/session":
					if r.Header.Get("X-Goog-Api-Key") != "" || r.Header.Get("X-Goog-Upload-Command") != "upload, finalize" {
						t.Errorf("upload headers = %v", r.Header)
					}
					body, _ := io.ReadAll(r.Body)
					if string(body) != "data" {
						t.Errorf("body = %s", body)
					}
					io.WriteString(w, `{"file":{"name":"files/abc","uri":"gs://file","state":"PROCESSING","mimeType":"text/plain"}}`)
				case "/v1beta/files/abc":
					if r.Header.Get("X-Goog-Api-Key") != "key" {
						t.Error("missing poll auth")
					}
					io.WriteString(w, `{"name":"files/abc","uri":"gs://file","state":"`+state+`","mimeType":"text/plain"}`)
				default:
					t.Errorf("unexpected path %s", r.URL.Path)
					w.WriteHeader(404)
				}
			}))
			defer server.Close()
			files := New(Options{APIKey: "key", BaseURL: server.URL + "/v1beta"}).Files()
			if _, ok := files.(model.FileDeleter); ok {
				t.Fatal("upload-only capability exposes deletion")
			}
			result, err := files.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), Filename: "test.txt", MediaType: "text/plain", ProviderOptions: map[string]any{"google": map[string]any{"displayName": "test", "pollIntervalMs": 1, "pollTimeoutMs": 50}}})
			if state == "ACTIVE" {
				if err != nil {
					t.Fatal(err)
				}
				if result.ProviderReference["google"] != "gs://file" || len(result.Warnings) != 1 {
					t.Fatalf("result = %+v", result)
				}
			} else if err == nil {
				t.Fatal("expected processing error")
			} else if state == "PROCESSING" && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestFilesMissingUploadURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()
	_, err := New(Options{BaseURL: server.URL + "/v1beta"}).Files().UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("data"), MediaType: "text/plain"})
	if err == nil || !strings.Contains(err.Error(), "upload URL") {
		t.Fatalf("error = %v", err)
	}
}
