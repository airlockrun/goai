package mistral

import (
	"context"
	"encoding/base64"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type MistralSpeechModel struct {
	id       string
	provider *Provider
}

func (m *MistralSpeechModel) ID() string       { return m.id }
func (m *MistralSpeechModel) Provider() string { return "mistral" }
func (m *MistralSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	format := opts.OutputFormat
	if format == "" {
		format = "mp3"
	}
	var warnings []stream.Warning
	switch format {
	case "pcm", "wav", "mp3", "flac", "opus":
	default:
		warnings = append(warnings, stream.UnsupportedWarning("outputFormat", "using mp3"))
		format = "mp3"
	}
	if opts.Speed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("speed", ""))
	}
	body := map[string]any{"model": m.id, "input": opts.Text, "response_format": format, "stream": false}
	if ref, ok := opts.ProviderOptions["refAudio"]; ok {
		body["ref_audio"] = ref
	} else if opts.Voice != "" {
		body["voice_id"] = opts.Voice
	}
	var result struct {
		Audio string `json:"audio_data"`
	}
	headers, err := m.provider.compat.DoJSON(ctx, "/audio/speech", body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	audio, err := base64.StdEncoding.DecodeString(result.Audio)
	if err != nil {
		return nil, err
	}
	mediaType := "audio/" + format
	if format == "mp3" {
		mediaType = "audio/mpeg"
	}
	return &model.SpeechResult{Audio: audio, MimeType: mediaType, Warnings: warnings, Response: model.SpeechResponse{Model: m.id, Headers: headers}}, nil
}
