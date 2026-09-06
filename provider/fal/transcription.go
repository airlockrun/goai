package fal

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/openaicompat"
	goairesponse "github.com/airlockrun/goai/response"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type FalTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *FalTranscriptionModel) ID() string       { return m.id }
func (m *FalTranscriptionModel) Provider() string { return "fal" }
func (m *FalTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	audioURL := opts.AudioURL
	if audioURL == "" {
		audio := opts.Audio
		if opts.AudioReader != nil {
			var err error
			audio, err = io.ReadAll(opts.AudioReader)
			if err != nil {
				return nil, err
			}
		}
		audioURL = "data:" + opts.MimeType + ";base64," + base64.StdEncoding.EncodeToString(audio)
	}
	body := map[string]any{"audio_url": audioURL, "task": "transcribe", "diarize": true, "chunk_level": "word"}
	if opts.Language != "" {
		body["language"] = opts.Language
	}
	for key, wire := range map[string]string{"language": "language", "version": "version", "batchSize": "batch_size", "numSpeakers": "num_speakers", "diarize": "diarize", "chunkLevel": "chunk_level"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			body[wire] = value
		}
	}
	path := "/fal-ai/" + strings.TrimPrefix(m.id, "fal-ai/")
	client := openaicompat.New(openaicompat.Options{ProviderID: m.Provider(), BaseURL: strings.TrimRight(m.provider.opts.BaseURL, "/"), APIKey: m.provider.opts.APIKey, Headers: m.provider.opts.Headers, AuthPrefix: "Key "})
	var job struct {
		ID string `json:"request_id"`
	}
	if _, err := client.DoJSON(ctx, path, body, &job, opts.Headers); err != nil {
		return nil, err
	}
	if job.ID == "" {
		return nil, errors.New("fal transcription response has no request_id")
	}
	endpoint := strings.TrimRight(m.provider.opts.BaseURL, "/") + path + "/requests/" + url.PathEscape(job.ID)
	for {
		req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Key "+m.provider.opts.APIKey)
		for k, v := range m.provider.opts.Headers {
			req.Header.Set(k, v)
		}
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		var result struct {
			Detail    string   `json:"detail"`
			Text      string   `json:"text"`
			Languages []string `json:"inferred_languages"`
			Chunks    []struct {
				Text      string    `json:"text"`
				Timestamp []float64 `json:"timestamp"`
			} `json:"chunks"`
		}
		if err := json.Unmarshal(data, &result); err != nil {
			return nil, err
		}
		if result.Detail != "Request is still in progress" {
			if resp.StatusCode != http.StatusOK {
				return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: string(data), URL: endpoint, StatusCode: resp.StatusCode, ResponseBody: string(data)})
			}
			out := &model.TranscriptionResult{Text: result.Text, Response: model.TranscriptionResponse{Model: m.id, Headers: goairesponse.ExtractResponseHeaders(resp)}}
			if len(result.Languages) > 0 {
				out.Language = result.Languages[0]
			}
			for i, chunk := range result.Chunks {
				part := model.TranscriptionSegment{ID: i, Text: chunk.Text}
				if len(chunk.Timestamp) > 0 {
					part.Start = chunk.Timestamp[0]
				}
				if len(chunk.Timestamp) > 1 {
					part.End = chunk.Timestamp[1]
				}
				out.Segments = append(out.Segments, part)
			}
			if len(out.Segments) > 0 {
				duration := out.Segments[len(out.Segments)-1].End
				out.Duration = &duration
			}
			return out, nil
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}
