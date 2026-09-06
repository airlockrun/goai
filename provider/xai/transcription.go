package xai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	goairesponse "github.com/airlockrun/goai/response"
)

type XaiTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *XaiTranscriptionModel) ID() string       { return m.id }
func (m *XaiTranscriptionModel) Provider() string { return "xai" }
func (m *XaiTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	fields := map[string][]string{}
	if opts.Language != "" {
		fields["language"] = []string{opts.Language}
	}
	for key, wire := range map[string]string{"audioFormat": "audio_format", "sampleRate": "sample_rate", "language": "language", "format": "format", "multichannel": "multichannel", "channels": "channels", "diarize": "diarize", "fillerWords": "filler_words"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			fields[wire] = []string{fmt.Sprint(value)}
		}
	}
	if value, ok := opts.ProviderOptions["keyterm"]; ok {
		switch terms := value.(type) {
		case string:
			fields["keyterm"] = []string{terms}
		case []string:
			fields["keyterm"] = terms
		case []any:
			for _, term := range terms {
				value, ok := term.(string)
				if !ok {
					return nil, errors.New("keyterm entries must be strings")
				}
				fields["keyterm"] = append(fields["keyterm"], value)
			}
		default:
			return nil, errors.New("keyterm must be a string or string array")
		}
	}
	resp, err := m.provider.compat.AudioForm(ctx, "/stt", fields, opts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Text     string   `json:"text"`
		Language string   `json:"language"`
		Duration *float64 `json:"duration"`
		Words    []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	out := &model.TranscriptionResult{Text: result.Text, Language: result.Language, Duration: result.Duration, Response: model.TranscriptionResponse{Model: m.id, Headers: goairesponse.ExtractResponseHeaders(resp)}}
	for i, word := range result.Words {
		out.Segments = append(out.Segments, model.TranscriptionSegment{ID: i, Text: word.Text, Start: word.Start, End: word.End})
	}
	return out, nil
}
