package google

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type FilesOptions struct {
	DisplayName    string `json:"displayName"`
	PollIntervalMS *int64 `json:"pollIntervalMs"`
	PollTimeoutMS  *int64 `json:"pollTimeoutMs"`
}

type files struct{ client *internal.FilesClient }

func (p *Provider) Files() model.Files {
	headers := map[string]string{"x-goog-api-key": p.opts.APIKey}
	headers = internal.FilesHeaders(headers, p.opts.Headers)
	return &files{client: &internal.FilesClient{ProviderID: "google", BaseURL: p.opts.BaseURL, Headers: headers}}
}

func (f *files) Provider() string { return "google" }

func (f *files) UploadFile(ctx context.Context, opts *model.UploadFileOptions) (*model.UploadFileResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts == nil || opts.Data == nil || opts.MediaType == "" {
		return nil, errors.New("file data and media type are required")
	}
	settings, err := internal.ParseFilesOptions[FilesOptions](opts.ProviderOptions, "google")
	if err != nil {
		return nil, err
	}
	interval, timeout := 2*time.Second, 5*time.Minute
	for _, setting := range []struct {
		value *int64
		dest  *time.Duration
	}{{settings.PollIntervalMS, &interval}, {settings.PollTimeoutMS, &timeout}} {
		if setting.value != nil {
			if *setting.value <= 0 || *setting.value > int64((1<<63-1)/time.Millisecond) {
				return nil, errors.New("google file polling durations must be positive milliseconds")
			}
			*setting.dest = time.Duration(*setting.value) * time.Millisecond
		}
	}
	data, err := io.ReadAll(opts.Data)
	if err != nil {
		return nil, err
	}
	metadata := map[string]any{}
	if settings.DisplayName != "" {
		metadata["display_name"] = settings.DisplayName
	}
	body, err := json.Marshal(map[string]any{"file": metadata})
	if err != nil {
		return nil, err
	}
	headers := make(map[string]string, len(opts.Headers)+5)
	for k, v := range opts.Headers {
		headers[http.CanonicalHeaderKey(k)] = v
	}
	headers["X-Goog-Upload-Protocol"] = "resumable"
	headers["X-Goog-Upload-Command"] = "start"
	headers["X-Goog-Upload-Header-Content-Length"] = strconv.Itoa(len(data))
	headers["X-Goog-Upload-Header-Content-Type"] = opts.MediaType
	headers["Content-Type"] = "application/json"
	base := strings.TrimRight(f.client.BaseURL, "/")
	resp, err := f.client.Request(ctx, http.MethodPost, strings.TrimSuffix(base, "/v1beta")+"/upload/v1beta/files", headers, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	uploadURL := resp.Header.Get("X-Goog-Upload-Url")
	resp.Body.Close()
	parsed, err := url.Parse(uploadURL)
	if err != nil || parsed.Host == "" || parsed.User != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") {
		return nil, errors.New("google files: missing or invalid upload URL")
	}
	// The upload URL is a capability URL; provider credentials must not follow it.
	uploader := internal.FilesClient{ProviderID: "google"}
	resp, err = uploader.Request(ctx, http.MethodPost, uploadURL, map[string]string{"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"}, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	var upload struct {
		File json.RawMessage `json:"file"`
	}
	err = json.NewDecoder(resp.Body).Decode(&upload)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}
	raw := upload.File
	pollCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for {
		var file struct {
			Name     string `json:"name"`
			URI      string `json:"uri"`
			State    string `json:"state"`
			MimeType string `json:"mimeType"`
		}
		if err := json.Unmarshal(raw, &file); err != nil {
			return nil, err
		}
		if file.Name == "" || file.URI == "" || file.State == "" {
			return nil, errors.New("google files: incomplete file response")
		}
		if file.State == "FAILED" {
			return nil, fmt.Errorf("google file processing failed: %s", file.Name)
		}
		if file.State != "PROCESSING" {
			if file.State != "ACTIVE" {
				return nil, fmt.Errorf("google files: unknown file state %q", file.State)
			}
			var providerMetadata map[string]any
			if err := json.Unmarshal(raw, &providerMetadata); err != nil {
				return nil, err
			}
			mediaType := file.MimeType
			if mediaType == "" {
				mediaType = opts.MediaType
			}
			result := &model.UploadFileResult{ProviderReference: model.ProviderReference{"google": file.URI}, MediaType: mediaType, ProviderMetadata: map[string]any{"google": providerMetadata}}
			if opts.Filename != "" {
				result.Warnings = []stream.Warning{stream.UnsupportedWarning("filename", "")}
			}
			return result, nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-timer.C:
		case <-pollCtx.Done():
			timer.Stop()
			return nil, pollCtx.Err()
		}
		id := strings.TrimPrefix(file.Name, "files/")
		if id == "" || strings.Contains(id, "/") || id == "." || id == ".." {
			return nil, errors.New("google files: invalid file resource name")
		}
		resp, err := f.client.Request(pollCtx, http.MethodGet, base+"/files/"+url.PathEscape(id), opts.Headers, nil)
		if err != nil {
			return nil, err
		}
		err = json.NewDecoder(resp.Body).Decode(&raw)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
	}
}
