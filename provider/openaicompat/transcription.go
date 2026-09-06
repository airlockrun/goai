package openaicompat

import (
	"bytes"
	"context"
	"errors"
	"github.com/airlockrun/goai/model"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"sort"
)

// AudioForm sends multipart transcription input with the file as the final field.
// Remote audio is deliberately not downloaded by this helper.
func (p *Provider) AudioForm(ctx context.Context, path string, fields map[string][]string, opts model.TranscribeCallOptions) (*http.Response, error) {
	if opts.AudioURL != "" {
		return nil, errors.New("audio URL is unsupported by this multipart endpoint; provide audio bytes or a reader")
	}
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		for _, value := range fields[key] {
			if err := w.WriteField(key, value); err != nil {
				return nil, err
			}
		}
	}
	filename := opts.Filename
	if filename == "" {
		filename = "audio"
		if extensions, _ := mime.ExtensionsByType(opts.MimeType); len(extensions) > 0 {
			filename += extensions[0]
		}
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{"name": "file", "filename": filename}))
	mediaType := opts.MimeType
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	header.Set("Content-Type", mediaType)
	part, err := w.CreatePart(header)
	if err != nil {
		return nil, err
	}
	reader := opts.AudioReader
	if reader == nil {
		reader = bytes.NewReader(opts.Audio)
	}
	if _, err := io.Copy(part, reader); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return p.Do(ctx, path, w.FormDataContentType(), &body, opts.Headers)
}
