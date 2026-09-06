package google

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"strconv"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type GoogleTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *GoogleTranscriptionModel) ID() string       { return m.id }
func (m *GoogleTranscriptionModel) Provider() string { return "google" }
func (m *GoogleTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	if strings.HasSuffix(m.id, "-live") {
		return nil, fmt.Errorf("%w: live models require streaming transcription", goaierrors.ErrUnsupported)
	}
	if opts.AudioURL != "" {
		return nil, fmt.Errorf("%w: transcription requires inline audio", goaierrors.ErrUnsupported)
	}
	audio := opts.Audio
	if opts.AudioReader != nil {
		var err error
		audio, err = io.ReadAll(opts.AudioReader)
		if err != nil {
			return nil, err
		}
	}
	config := map[string]any{}
	for key, wire := range map[string]string{"languageCodes": "language_codes", "customVocabulary": "custom_vocabulary"} {
		if value, ok := opts.ProviderOptions[key]; ok {
			config[wire] = value
		}
	}
	if opts.Language != "" {
		if _, ok := config["language_codes"]; !ok {
			config["language_codes"] = []string{opts.Language}
		}
	}
	mode := map[string]any{}
	if value, ok := opts.ProviderOptions["mode"].(string); ok {
		mode["type"] = strings.ToLower(value)
	}
	if opts.ProviderOptions["diarization"] == true {
		mode["diarization_mode"] = "speaker"
	}
	if opts.ProviderOptions["wordTimestamp"] == true {
		mode["timestamp_granularities"] = []string{"word"}
	}
	if len(mode) > 0 {
		if _, ok := mode["type"]; !ok {
			mode["type"] = "verbatim"
		}
		config["mode"] = mode
	}
	request := map[string]any{"model": m.id, "input": []any{map[string]any{"type": "audio", "data": base64.StdEncoding.EncodeToString(audio), "mime_type": opts.MimeType}}}
	if len(config) > 0 {
		request["generation_config"] = map[string]any{"transcription_config": config}
	}
	var response struct {
		Steps []struct {
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Annotations []struct {
					Type  string `json:"type"`
					Text  string `json:"text"`
					Start string `json:"start_offset"`
					End   string `json:"end_offset"`
				} `json:"annotations"`
			} `json:"content"`
		} `json:"steps"`
	}
	headers, err := m.provider.post(ctx, m.id, "interactions", request, opts.Headers, &response)
	if err != nil {
		return nil, err
	}
	result := &model.TranscriptionResult{Response: model.TranscriptionResponse{Model: m.id, Headers: headers}}
	if opts.Prompt != "" {
		result.Warnings = append(result.Warnings, stream.UnsupportedWarning("prompt", "Use customVocabulary."))
	}
	for _, step := range response.Steps {
		for _, part := range step.Content {
			if part.Type != "text" {
				continue
			}
			result.Text += part.Text
			for _, word := range part.Annotations {
				if word.Type != "word_info" {
					continue
				}
				start, e1 := strconv.ParseFloat(strings.TrimSuffix(word.Start, "s"), 64)
				end, e2 := strconv.ParseFloat(strings.TrimSuffix(word.End, "s"), 64)
				if e1 == nil && e2 == nil {
					result.Segments = append(result.Segments, model.TranscriptionSegment{ID: len(result.Segments), Text: word.Text, Start: start, End: end})
				}
			}
		}
	}
	return result, nil
}
