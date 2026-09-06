package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strings"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
)

// FilesClient shares the multipart and file-resource HTTP protocol.
type FilesClient struct {
	ProviderID   string
	BaseURL      string
	Headers      map[string]string
	UploadFields func(map[string]any) ([][2]string, error)
}

func (c *FilesClient) Provider() string { return c.ProviderID }

// FilesHeaders merges headers case-insensitively, with configured values last.
func FilesHeaders(defaults, configured map[string]string) map[string]string {
	headers := make(map[string]string, len(defaults)+len(configured))
	for k, v := range defaults {
		headers[http.CanonicalHeaderKey(k)] = v
	}
	for k, v := range configured {
		headers[http.CanonicalHeaderKey(k)] = v
	}
	return headers
}

// Request returns an owned response body on success. It never retries readers.
func (c *FilesClient) Request(ctx context.Context, method, target string, headers map[string]string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return nil, err
	}
	for k, v := range c.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	// File requests carry credentials and potentially non-replayable uploads.
	// Do not forward either to a redirect target or change the request method.
	client := *http.DefaultClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		data, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if readErr != nil {
			return nil, readErr
		}
		return nil, &goaierrors.APICallError{URL: target, StatusCode: resp.StatusCode, ResponseBody: string(data), Message: fmt.Sprintf("%s files: HTTP %d: %s", c.ProviderID, resp.StatusCode, data), IsRetryable: resp.StatusCode == 429 || resp.StatusCode >= 500}
	}
	return resp, nil
}

func (c *FilesClient) UploadFile(ctx context.Context, opts *model.UploadFileOptions) (*model.UploadFileResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts == nil || opts.Data == nil || opts.MediaType == "" {
		return nil, errors.New("file data and media type are required")
	}
	if _, _, err := mime.ParseMediaType(opts.MediaType); err != nil || strings.ContainsAny(opts.MediaType, "\r\n") {
		return nil, errors.New("invalid file media type")
	}
	var fields [][2]string
	if c.UploadFields != nil {
		var err error
		fields, err = c.UploadFields(opts.ProviderOptions)
		if err != nil {
			return nil, err
		}
	}
	// Only the multipart framing is buffered; the file reader streams directly.
	var prefix bytes.Buffer
	w := multipart.NewWriter(&prefix)
	for _, field := range fields {
		if err := w.WriteField(field[0], field[1]); err != nil {
			return nil, err
		}
	}
	filename := opts.Filename
	if filename == "" {
		filename = "blob"
	}
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{"name": "file", "filename": filename}))
	h.Set("Content-Type", opts.MediaType)
	if _, err := w.CreatePart(h); err != nil {
		return nil, err
	}
	framing := append([]byte(nil), prefix.Bytes()...)
	prefix.Reset()
	if err := w.Close(); err != nil {
		return nil, err
	}
	headers := make(map[string]string, len(opts.Headers)+1)
	for k, v := range opts.Headers {
		if !strings.EqualFold(k, "Content-Type") {
			headers[k] = v
		}
	}
	headers["Content-Type"] = w.FormDataContentType()
	resp, err := c.Request(ctx, http.MethodPost, strings.TrimRight(c.BaseURL, "/")+"/files", headers, io.MultiReader(bytes.NewReader(framing), opts.Data, bytes.NewReader(prefix.Bytes())))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	result, err := c.DecodeMetadata(resp.Body)
	if err != nil {
		return nil, err
	}
	if result.Filename == "" {
		result.Filename = opts.Filename
	}
	if result.MediaType == "" {
		result.MediaType = opts.MediaType
	}
	return result, nil
}

func (c *FilesClient) fileURL(opts *model.FileOptions) (string, error) {
	if opts == nil {
		return "", errors.New("file reference is required")
	}
	id, ok := opts.File[c.ProviderID].(string)
	if !ok || strings.TrimSpace(id) == "" {
		return "", fmt.Errorf("file reference is missing a %q file id", c.ProviderID)
	}
	segment := url.PathEscape(id)
	if id == "." {
		segment = "%252E"
	} else if id == ".." {
		segment = "%252E%252E"
	}
	return strings.TrimRight(c.BaseURL, "/") + "/files/" + segment, nil
}

func (c *FilesClient) GetFileMetadata(ctx context.Context, opts *model.FileOptions) (*model.FileMetadata, error) {
	target, err := c.fileURL(opts)
	if err != nil {
		return nil, err
	}
	resp, err := c.Request(ctx, http.MethodGet, target, opts.Headers, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return c.DecodeMetadata(resp.Body)
}

func (c *FilesClient) DownloadFile(ctx context.Context, opts *model.FileOptions) (*model.DownloadFileResult, error) {
	target, err := c.fileURL(opts)
	if err != nil {
		return nil, err
	}
	resp, err := c.Request(ctx, http.MethodGet, target+"/content", opts.Headers, nil)
	if err != nil {
		return nil, err
	}
	return &model.DownloadFileResult{Content: resp.Body, MediaType: strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])}, nil
}

func (c *FilesClient) DeleteFile(ctx context.Context, opts *model.FileOptions) (*model.DeleteFileResult, error) {
	target, err := c.fileURL(opts)
	if err != nil {
		return nil, err
	}
	resp, err := c.Request(ctx, http.MethodDelete, target, opts.Headers, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var wire struct {
		ID      string `json:"id"`
		Deleted *bool  `json:"deleted"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return nil, err
	}
	if wire.ID == "" || wire.Deleted == nil {
		return nil, fmt.Errorf("%w: missing file deletion id or deleted flag", goaierrors.ErrInvalidResponse)
	}
	return &model.DeleteFileResult{ProviderReference: model.ProviderReference{c.ProviderID: wire.ID}, Deleted: *wire.Deleted}, nil
}

func (c *FilesClient) DecodeMetadata(r io.Reader) (*model.FileMetadata, error) {
	var wire struct {
		ID        string          `json:"id"`
		Filename  string          `json:"filename"`
		MediaType string          `json:"mime_type"`
		Bytes     *int64          `json:"bytes"`
		SizeBytes *int64          `json:"size_bytes"`
		CreatedAt json.RawMessage `json:"created_at"`
		ExpiresAt *int64          `json:"expires_at"`
	}
	var raw json.RawMessage
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, err
	}
	if strings.TrimSpace(wire.ID) == "" {
		return nil, fmt.Errorf("%w: missing file id", goaierrors.ErrInvalidResponse)
	}
	var metadata map[string]any
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil, err
	}
	for _, pair := range [][2]string{{"created_at", "createdAt"}, {"expires_at", "expiresAt"}, {"mime_type", "mimeType"}, {"size_bytes", "sizeBytes"}} {
		if value, ok := metadata[pair[0]]; ok {
			metadata[pair[1]] = value
			delete(metadata, pair[0])
		}
	}
	delete(metadata, "id")
	result := &model.FileMetadata{ProviderReference: model.ProviderReference{c.ProviderID: wire.ID}, Filename: wire.Filename, MediaType: wire.MediaType, ByteSize: wire.Bytes, ProviderMetadata: map[string]any{c.ProviderID: metadata}}
	if result.ByteSize == nil {
		result.ByteSize = wire.SizeBytes
	}
	if result.ByteSize != nil && *result.ByteSize < 0 {
		return nil, fmt.Errorf("%w: negative file size", goaierrors.ErrInvalidResponse)
	}
	if len(wire.CreatedAt) != 0 && string(wire.CreatedAt) != "null" {
		var created time.Time
		if wire.CreatedAt[0] == '"' {
			if err := json.Unmarshal(wire.CreatedAt, &created); err != nil {
				return nil, err
			}
		} else {
			var seconds int64
			if err := json.Unmarshal(wire.CreatedAt, &seconds); err != nil {
				return nil, err
			}
			created = time.Unix(seconds, 0)
		}
		result.CreatedAt = &created
	}
	if wire.ExpiresAt != nil {
		expires := time.Unix(*wire.ExpiresAt, 0)
		result.ExpiresAt = &expires
	}
	return result, nil
}

// ParseFilesOptions accepts provider-namespaced options, matching file references.
func ParseFilesOptions[T any](options map[string]any, providerID string) (T, error) {
	var result T
	value := options[providerID]
	if value == nil {
		return result, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return result, err
	}
	err = json.Unmarshal(raw, &result)
	return result, err
}
