package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	goaierrors "github.com/airlockrun/goai/errors"
	"io"
	"net/http"
)

// Do sends a modality request to a path relative to the configured base URL.
// The caller owns the successful response body.
func (p *Provider) Do(ctx context.Context, path, contentType string, body io.Reader, headers map[string]string) (*http.Response, error) {
	url := p.opts.BaseURL + path
	req, err := http.NewRequestWithContext(ctx, "POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	if p.opts.APIKey != "" {
		req.Header.Set(p.opts.AuthHeader, p.opts.AuthPrefix+p.opts.APIKey)
	}
	for k, v := range p.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	client := p.opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: err.Error(), URL: url, Cause: err, IsRetryable: true, IsRetryableSet: true})
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorResponseBodyBytes))
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: string(data), URL: url, StatusCode: resp.StatusCode, ResponseHeaders: flattenHeaders(resp.Header), ResponseBody: string(data)})
	}
	return resp, nil
}

// DoJSON sends and decodes a JSON modality request, returning response headers.
func (p *Provider) DoJSON(ctx context.Context, path string, body, result any, headers map[string]string) (map[string]string, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := p.Do(ctx, path, "application/json", bytes.NewReader(data), headers)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
		return nil, err
	}
	return flattenHeaders(resp.Header), nil
}
