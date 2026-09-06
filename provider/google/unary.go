package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	goaierrors "github.com/airlockrun/goai/errors"
	"io"
	"net/http"
	"strings"
)

func modelPath(id string) string {
	if strings.Contains(id, "/") {
		return id
	}
	return "models/" + id
}

func (p *Provider) post(ctx context.Context, id, action string, body any, headers map[string]string, result any) (map[string]string, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(p.opts.BaseURL, "/") + "/" + modelPath(id) + ":" + action
	if action == "interactions" {
		url = strings.TrimRight(p.opts.BaseURL, "/") + "/interactions"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.opts.APIKey != "" {
		req.Header.Set("x-goog-api-key", p.opts.APIKey)
	}
	for k, v := range p.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: "Google request failed", URL: url, Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true})
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: string(raw), URL: url, StatusCode: resp.StatusCode, ResponseBody: string(raw), ResponseHeaders: responseHeaders(resp.Header)})
	}
	var envelope struct {
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error != nil {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: envelope.Error.Message, StatusCode: envelope.Error.Code, URL: url, ResponseBody: string(raw)})
	}
	if err := json.Unmarshal(raw, result); err != nil {
		return nil, fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)
	}
	return responseHeaders(resp.Header), nil
}
