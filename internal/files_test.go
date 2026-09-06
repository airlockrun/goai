package internal

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
)

func TestFilesClientLifecycle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("X-Test") != "call" {
			t.Errorf("headers = %v", r.Header)
		}
		switch {
		case r.Method == http.MethodPost:
			reader, err := r.MultipartReader()
			if err != nil {
				t.Error(err)
				return
			}
			for i, want := range []string{"purpose", "file"} {
				part, err := reader.NextPart()
				if err != nil {
					t.Error(err)
					return
				}
				data, err := io.ReadAll(part)
				if err != nil {
					t.Error(err)
					return
				}
				if part.FormName() != want {
					t.Errorf("part %d = %q", i, part.FormName())
				}
				if want == "file" && (string(data) != "hello" || part.FileName() != "test.txt" || part.Header.Get("Content-Type") != "text/plain") {
					t.Errorf("file = %s, %v", data, part.Header)
				}
			}
			if _, err := reader.NextPart(); err != io.EOF {
				t.Errorf("multipart end = %v", err)
			}
			io.WriteString(w, `{"id":"f/1","filename":"test.txt","bytes":5,"created_at":123,"expires_at":456}`)
		case r.Method == http.MethodDelete:
			io.WriteString(w, `{"id":"f/1","deleted":false}`)
		case strings.HasSuffix(r.URL.Path, "/content"):
			if r.URL.EscapedPath() != "/files/f%2F1/content" {
				t.Errorf("path = %s", r.URL.EscapedPath())
			}
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			io.WriteString(w, "hello")
		default:
			if r.URL.EscapedPath() != "/files/f%2F1" {
				t.Errorf("path = %s", r.URL.EscapedPath())
			}
			io.WriteString(w, `{"id":"f/1","bytes":5}`)
		}
	}))
	defer server.Close()
	c := &FilesClient{ProviderID: "test", BaseURL: server.URL, Headers: map[string]string{"Authorization": "Bearer key", "X-Test": "provider"}, UploadFields: func(map[string]any) ([][2]string, error) { return [][2]string{{"purpose", "assistants"}}, nil }}
	ctx := context.Background()
	headers := map[string]string{"X-Test": "call"}
	upload, err := c.UploadFile(ctx, &model.UploadFileOptions{Data: strings.NewReader("hello"), MediaType: "text/plain", Filename: "test.txt", Headers: headers})
	if err != nil {
		t.Fatal(err)
	}
	if upload.ProviderReference["test"] != "f/1" || *upload.ByteSize != 5 || upload.CreatedAt.Unix() != 123 || upload.ExpiresAt.Unix() != 456 {
		t.Fatalf("upload = %+v", upload)
	}
	options := &model.FileOptions{File: upload.ProviderReference, Headers: headers}
	if _, err := c.GetFileMetadata(ctx, options); err != nil {
		t.Fatal(err)
	}
	download, err := c.DownloadFile(ctx, options)
	if err != nil {
		t.Fatal(err)
	}
	defer download.Content.Close()
	data, err := io.ReadAll(download.Content)
	if err != nil || string(data) != "hello" || download.MediaType != "text/plain" {
		t.Fatalf("download = %q, %v, %v", data, download, err)
	}
	deleted, err := c.DeleteFile(ctx, options)
	if err != nil || deleted.Deleted {
		t.Fatalf("delete = %v, %v", deleted, err)
	}
}

func TestFilesClientErrors(t *testing.T) {
	for _, tc := range []struct {
		name, response string
		status         int
	}{
		{"api error", `{"error":{"message":"no"}}`, 429},
		{"bad json", `{`, 200},
		{"missing id", `{}`, 200},
		{"bad size", `{"id":"f","bytes":-1}`, 200},
		{"bad date", `{"id":"f","created_at":"bad"}`, 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tc.status); io.WriteString(w, tc.response) }))
			defer server.Close()
			c := &FilesClient{ProviderID: "test", BaseURL: server.URL}
			_, err := c.GetFileMetadata(context.Background(), &model.FileOptions{File: model.ProviderReference{"test": "f"}})
			if err == nil {
				t.Fatal("expected error")
			}
			if tc.status == 429 {
				var apiErr *goaierrors.APICallError
				if !errors.As(err, &apiErr) || !apiErr.IsRetryable {
					t.Fatalf("error = %v", err)
				}
			}
		})
	}
	for _, ref := range []model.ProviderReference{nil, {"test": " "}, {"other": "f"}, {"test": 1}} {
		c := &FilesClient{ProviderID: "test"}
		if _, err := c.GetFileMetadata(context.Background(), &model.FileOptions{File: ref}); err == nil {
			t.Fatalf("accepted %v", ref)
		}
	}
	c := &FilesClient{ProviderID: "test", BaseURL: "http://example.com"}
	for _, tc := range []struct{ id, want string }{{".", "%252E"}, {"..", "%252E%252E"}, {"a?b#c", "a%3Fb%23c"}} {
		t.Run(tc.id, func(t *testing.T) {
			got, err := c.fileURL(&model.FileOptions{File: model.ProviderReference{"test": tc.id}})
			if err != nil || got != c.BaseURL+"/files/"+tc.want {
				t.Fatalf("url = %s, %v", got, err)
			}
		})
	}
}

func TestFilesClientCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := &FilesClient{ProviderID: "test", BaseURL: "http://example.com"}
	_, err := c.UploadFile(ctx, &model.UploadFileOptions{Data: strings.NewReader("test"), MediaType: "text/plain"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}

func TestFilesHeaders(t *testing.T) {
	defaults := map[string]string{"Authorization": "default", "X-Test": "kept"}
	got := FilesHeaders(defaults, map[string]string{"authorization": "configured"})
	if len(got) != 2 || got["Authorization"] != "configured" || got["X-Test"] != "kept" || defaults["Authorization"] != "default" {
		t.Fatalf("headers = %v, defaults = %v", got, defaults)
	}
}

func TestFilesClientRejectsRedirects(t *testing.T) {
	for _, status := range []int{http.StatusFound, http.StatusTemporaryRedirect} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Errorf("redirect followed with headers %v", r.Header)
				io.WriteString(w, `{"id":"leaked"}`)
			}))
			defer target.Close()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, target.URL, status)
			}))
			defer server.Close()
			c := &FilesClient{ProviderID: "test", BaseURL: server.URL, Headers: map[string]string{"x-api-key": "secret", "x-goog-api-key": "secret"}}
			_, err := c.GetFileMetadata(context.Background(), &model.FileOptions{File: model.ProviderReference{"test": "f"}})
			var apiErr *goaierrors.APICallError
			if !errors.As(err, &apiErr) || apiErr.StatusCode != status {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestFilesClientRejectsMultipartHeaderInjection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("invalid upload reached server")
		io.WriteString(w, `{"id":"f"}`)
	}))
	defer server.Close()
	c := &FilesClient{ProviderID: "test", BaseURL: server.URL}
	for _, mediaType := range []string{"text/plain\r\nX-Injected: yes", "text/plain\nX-Injected: yes", "not a media type"} {
		t.Run(mediaType, func(t *testing.T) {
			_, err := c.UploadFile(context.Background(), &model.UploadFileOptions{Data: strings.NewReader("hello"), MediaType: mediaType})
			if err == nil {
				t.Fatal("accepted invalid multipart media type")
			}
		})
	}
}
