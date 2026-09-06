package goai_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/airlockrun/goai"
	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
)

type uploadOnlyFiles struct{ data, mediaType string }

func (f *uploadOnlyFiles) Provider() string { return "test" }
func (f *uploadOnlyFiles) UploadFile(ctx context.Context, opts *model.UploadFileOptions) (*model.UploadFileResult, error) {
	data, err := io.ReadAll(opts.Data)
	f.data = string(data)
	f.mediaType = opts.MediaType
	return &model.UploadFileResult{ProviderReference: model.ProviderReference{"test": "file-1"}}, err
}

func TestFilesPublicUpload(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input goai.UploadFileInput
		valid bool
		want  string
	}{
		{"bytes", goai.UploadFileInput{Data: []byte("data")}, true, "data"},
		{"reader", goai.UploadFileInput{DataReader: strings.NewReader("data")}, true, "data"},
		{"base64", goai.UploadFileInput{DataBase64: "ZGF0YQ=="}, true, "data"},
		{"empty file", goai.UploadFileInput{Data: []byte{}}, true, ""},
		{"missing", goai.UploadFileInput{}, false, ""},
		{"ambiguous", goai.UploadFileInput{Data: []byte("data"), DataBase64: "ZGF0YQ=="}, false, ""},
		{"bad base64", goai.UploadFileInput{DataBase64: "%%%"}, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := &uploadOnlyFiles{}
			tc.input.Files, tc.input.MediaType = files, "text/plain"
			result, err := goai.UploadFile(context.Background(), tc.input)
			if !tc.valid {
				if err == nil {
					t.Fatal("expected validation error")
				}
				return
			}
			if err != nil || files.data != tc.want || result.ProviderReference["test"] != "file-1" {
				t.Fatalf("result = %v, data = %s, error = %v", result, files.data, err)
			}
		})
	}
}

func TestFilesOptionalCapabilities(t *testing.T) {
	input := goai.FileInput{Files: &uploadOnlyFiles{}, File: model.ProviderReference{"test": "file-1"}}
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"metadata", func() error { _, err := goai.GetFileMetadata(context.Background(), input); return err }},
		{"download", func() error { _, err := goai.DownloadFile(context.Background(), input); return err }},
		{"delete", func() error { _, err := goai.DeleteFile(context.Background(), input); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); !errors.Is(err, goaierrors.ErrUnsupported) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestFilesMediaTypeDetection(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input goai.UploadFileInput
		want  string
	}{
		{"text", goai.UploadFileInput{Data: []byte("hello")}, "text/plain"},
		{"pdf", goai.UploadFileInput{Data: []byte("%PDF-1.7")}, "application/pdf"},
		{"base64", goai.UploadFileInput{DataBase64: "ZGF0YQ=="}, "text/plain"},
		{"reader", goai.UploadFileInput{DataReader: strings.NewReader("hello")}, "application/octet-stream"},
		{"empty", goai.UploadFileInput{Data: []byte{}}, "application/octet-stream"},
		{"explicit", goai.UploadFileInput{Data: []byte("{}"), MediaType: "application/json"}, "application/json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := &uploadOnlyFiles{}
			tc.input.Files = files
			if _, err := goai.UploadFile(context.Background(), tc.input); err != nil {
				t.Fatal(err)
			}
			if files.mediaType != tc.want {
				t.Fatalf("media type = %q, want %q", files.mediaType, tc.want)
			}
		})
	}
}
