package elevenlabs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/openaicompat"
	goairesponse "github.com/airlockrun/goai/response"
	"github.com/airlockrun/goai/stream"
)

type ElevenLabsTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *ElevenLabsTranscriptionModel) ID() string       { return m.id }
func (m *ElevenLabsTranscriptionModel) Provider() string { return "elevenlabs" }
func (m *ElevenLabsTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	if m.id == "scribe_v2_realtime" {
		return nil, errors.New("scribe_v2_realtime requires a realtime transcription interface")
	}
	fields := map[string][]string{"model_id": {m.id}, "diarize": {"true"}}
	if opts.Language != "" {
		fields["language_code"] = []string{opts.Language}
	}
	for key, wire := range map[string]string{"languageCode": "language_code", "tagAudioEvents": "tag_audio_events", "numSpeakers": "num_speakers", "timestampsGranularity": "timestamps_granularity", "fileFormat": "file_format", "diarize": "diarize"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			fields[wire] = []string{fmt.Sprint(value)}
		}
	}
	headers := map[string]string{"xi-api-key": m.provider.opts.APIKey}
	for k, v := range m.provider.opts.Headers {
		headers[k] = v
	}
	client := openaicompat.New(openaicompat.Options{ProviderID: m.Provider(), BaseURL: m.provider.opts.BaseURL, Headers: headers})
	resp, err := client.AudioForm(ctx, "/speech-to-text", fields, opts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Text     string `json:"text"`
		Language string `json:"language_code"`
		Words    []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	out := &model.TranscriptionResult{Text: result.Text, Language: result.Language, Response: model.TranscriptionResponse{Model: m.id, Headers: goairesponse.ExtractResponseHeaders(resp)}}
	for i, word := range result.Words {
		out.Segments = append(out.Segments, model.TranscriptionSegment{ID: i, Text: word.Text, Start: word.Start, End: word.End})
	}
	if len(out.Segments) > 0 {
		end := out.Segments[len(out.Segments)-1].End
		out.Duration = &end
	}
	if _, ok := opts.ProviderOptions["streaming"]; ok {
		out.Warnings = append(out.Warnings, stream.UnsupportedWarning("streaming", "batch transcription does not support streaming options"))
	}
	return out, nil
}
