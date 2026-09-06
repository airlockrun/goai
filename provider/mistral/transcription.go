package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	goairesponse "github.com/airlockrun/goai/response"
)

type MistralTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *MistralTranscriptionModel) ID() string       { return m.id }
func (m *MistralTranscriptionModel) Provider() string { return "mistral" }
func (m *MistralTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	fields := map[string][]string{"model": {m.id}}
	if opts.Language != "" {
		fields["language"] = []string{opts.Language}
	}
	for key, wire := range map[string]string{"language": "language", "temperature": "temperature", "timestampGranularities": "timestamp_granularities", "diarize": "diarize", "contextBias": "context_bias"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			switch values := value.(type) {
			case []string:
				fields[wire] = values
			case []any:
				for _, item := range values {
					fields[wire] = append(fields[wire], fmt.Sprint(item))
				}
			default:
				fields[wire] = []string{fmt.Sprint(value)}
			}
		}
	}
	if len(fields["language"]) > 0 && len(fields["timestamp_granularities"]) > 0 {
		return nil, errors.New("Mistral language and timestampGranularities are mutually exclusive")
	}
	resp, err := m.provider.compat.AudioForm(ctx, "/audio/transcriptions", fields, opts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Model    string `json:"model"`
		Text     string `json:"text"`
		Language string `json:"language"`
		Segments []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"segments"`
		Usage struct {
			Seconds *float64 `json:"prompt_audio_seconds"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	out := &model.TranscriptionResult{Text: result.Text, Language: result.Language, Duration: result.Usage.Seconds, Response: model.TranscriptionResponse{Model: result.Model, Headers: goairesponse.ExtractResponseHeaders(resp)}}
	for i, part := range result.Segments {
		out.Segments = append(out.Segments, model.TranscriptionSegment{ID: i, Text: part.Text, Start: part.Start, End: part.End})
	}
	if out.Duration == nil && len(out.Segments) > 0 {
		end := out.Segments[len(out.Segments)-1].End
		out.Duration = &end
	}
	return out, nil
}
