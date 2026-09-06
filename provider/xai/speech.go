package xai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	goairesponse "github.com/airlockrun/goai/response"
	"github.com/airlockrun/goai/stream"
	"io"
)

type SpeechOptions struct {
	Language                 string            `json:"language,omitempty"`
	SampleRate               *int              `json:"sampleRate,omitempty"`
	BitRate                  *int              `json:"bitRate,omitempty"`
	OptimizeStreamingLatency *int              `json:"optimizeStreamingLatency,omitempty"`
	TextNormalization        *bool             `json:"textNormalization,omitempty"`
	WithTimestamps           *bool             `json:"withTimestamps,omitempty"`
	Replace                  map[string]string `json:"replace,omitempty"`
}
type XaiSpeechModel struct {
	id       string
	provider *Provider
}

func (m *XaiSpeechModel) ID() string       { return m.id }
func (m *XaiSpeechModel) Provider() string { return "xai" }
func (m *XaiSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	settings, err := provider.ParseProviderOptions[SpeechOptions](opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if settings.SampleRate != nil {
		switch *settings.SampleRate {
		case 8000, 16000, 22050, 24000, 44100, 48000:
		default:
			return nil, fmt.Errorf("unsupported xAI sample rate %d", *settings.SampleRate)
		}
	}
	if settings.BitRate != nil {
		switch *settings.BitRate {
		case 32000, 64000, 96000, 128000, 192000:
		default:
			return nil, fmt.Errorf("unsupported xAI bit rate %d", *settings.BitRate)
		}
	}
	if settings.OptimizeStreamingLatency != nil && (*settings.OptimizeStreamingLatency < 0 || *settings.OptimizeStreamingLatency > 2) {
		return nil, errors.New("xAI optimizeStreamingLatency must be between 0 and 2")
	}
	format := opts.OutputFormat
	if format == "" {
		format = "mp3"
	}
	var warnings []stream.Warning
	switch format {
	case "mp3", "wav", "pcm", "mulaw", "alaw":
	default:
		warnings = append(warnings, stream.UnsupportedWarning("outputFormat", "using mp3"))
		format = "mp3"
	}
	voice := opts.Voice
	if voice == "" {
		voice = "eve"
	}
	language := settings.Language
	if language == "" {
		language = "auto"
	}
	output := map[string]any{"codec": format}
	if settings.SampleRate != nil {
		output["sample_rate"] = *settings.SampleRate
	}
	if settings.BitRate != nil {
		if format == "mp3" {
			output["bit_rate"] = *settings.BitRate
		} else {
			warnings = append(warnings, stream.UnsupportedWarning("bitRate", "only mp3 supports bitRate"))
		}
	}
	body := map[string]any{"text": opts.Text, "voice_id": voice, "language": language, "output_format": output}
	if opts.Speed != nil {
		body["speed"] = *opts.Speed
	}
	if settings.OptimizeStreamingLatency != nil {
		body["optimize_streaming_latency"] = *settings.OptimizeStreamingLatency
	}
	if settings.TextNormalization != nil {
		body["text_normalization"] = *settings.TextNormalization
	}
	if settings.WithTimestamps != nil {
		body["with_timestamps"] = *settings.WithTimestamps
	}
	if settings.Replace != nil {
		body["replace"] = settings.Replace
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	resp, err := m.provider.compat.Do(ctx, "/tts", "application/json", bytes.NewReader(data), opts.Headers)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	out := &model.SpeechResult{Warnings: warnings, MimeType: resp.Header.Get("Content-Type"), Response: model.SpeechResponse{Model: m.id, ID: resp.Header.Get("X-Trace-Id"), Headers: goairesponse.ExtractResponseHeaders(resp)}}
	if settings.WithTimestamps != nil && *settings.WithTimestamps {
		var envelope struct {
			Audio       string   `json:"audio"`
			ContentType string   `json:"content_type"`
			Duration    *float64 `json:"duration"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
			return nil, err
		}
		out.Audio, err = base64.StdEncoding.DecodeString(envelope.Audio)
		out.MimeType = envelope.ContentType
		out.Duration = envelope.Duration
	} else {
		out.Audio, err = io.ReadAll(resp.Body)
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}
