package deepseek

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/model"
)

type FilesOptions struct {
	ExpiresAfter *int64 `json:"expiresAfter"`
}
type files struct{ client *internal.FilesClient }

func (p *Provider) Files() model.Files {
	headers := map[string]string{"Authorization": "Bearer " + p.compat.APIKey()}
	headers = internal.FilesHeaders(headers, p.compat.FileHeaders())
	return &files{client: &internal.FilesClient{ProviderID: "deepseek", BaseURL: p.compat.BaseURL(), Headers: headers, UploadFields: func(options map[string]any) ([][2]string, error) {
		opts, err := internal.ParseFilesOptions[FilesOptions](options, "deepseek")
		if err != nil {
			return nil, err
		}
		fields := [][2]string{{"purpose", "user_data"}}
		if opts.ExpiresAfter != nil {
			if *opts.ExpiresAfter < 3600 || *opts.ExpiresAfter > 2592000 {
				return nil, errors.New("deepseek expiresAfter must be between 3600 and 2592000 seconds")
			}
			fields = append(fields, [2]string{"expires_after[anchor]", "created_at"}, [2]string{"expires_after[seconds]", strconv.FormatInt(*opts.ExpiresAfter, 10)})
		}
		return fields, nil
	}}}
}

func (f *files) Provider() string { return "deepseek" }
func (f *files) UploadFile(ctx context.Context, opts *model.UploadFileOptions) (*model.UploadFileResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts == nil || opts.Data == nil {
		return nil, errors.New("file data is required")
	}
	if utf8.RuneCountInString(opts.Filename) > 512 {
		return nil, errors.New("deepseek filename must not exceed 512 characters")
	}
	if _, err := f.client.UploadFields(opts.ProviderOptions); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(opts.Data, (64<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > 64<<20 {
		return nil, errors.New("deepseek file must not exceed 64 MiB")
	}
	supported := func(mediaType string) bool {
		switch mediaType {
		case "image/gif", "image/jpeg", "image/jpg", "image/png", "image/webp":
			return true
		}
		return false
	}
	mediaType := strings.ToLower(strings.TrimSpace(strings.Split(opts.MediaType, ";")[0]))
	detected := http.DetectContentType(data)
	if detected != "application/octet-stream" && !strings.HasPrefix(detected, "text/plain") && !supported(detected) {
		return nil, errors.New("deepseek file content must be JPEG, PNG, GIF, or WebP")
	}
	if !supported(mediaType) {
		switch mediaType {
		case "", "application/binary", "application/octet-stream", "binary/octet-stream", "image", "image/*":
			ext := strings.ToLower(filepath.Ext(opts.Filename))
			if !supported(detected) && ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".gif" && ext != ".webp" {
				return nil, errors.New("deepseek file must identify a JPEG, PNG, GIF, or WebP image")
			}
		default:
			return nil, errors.New("deepseek file media type must be JPEG, PNG, GIF, or WebP")
		}
	}
	copy := *opts
	copy.Data = bytes.NewReader(data)
	if copy.MediaType == "" {
		copy.MediaType = "application/octet-stream"
	}
	return f.client.UploadFile(ctx, &copy)
}
