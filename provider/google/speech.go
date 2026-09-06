package google

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"mime"
	"strconv"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type GoogleSpeechModel struct {
	id       string
	provider *Provider
}

func (m *GoogleSpeechModel) ID() string       { return m.id }
func (m *GoogleSpeechModel) Provider() string { return "google" }
func (m *GoogleSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	voice := opts.Voice
	if voice == "" {
		voice = "Kore"
	}
	config := map[string]any{"voiceConfig": map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": voice}}}
	if multi, ok := opts.ProviderOptions["multiSpeakerVoiceConfig"]; ok {
		config = map[string]any{"multiSpeakerVoiceConfig": multi}
	}
	request := map[string]any{"contents": []any{map[string]any{"role": "user", "parts": []any{map[string]any{"text": opts.Text}}}}, "generationConfig": map[string]any{"responseModalities": []string{"AUDIO"}, "speechConfig": config}}
	var response geminiStreamChunk
	headers, err := m.provider.post(ctx, m.id, "generateContent", request, opts.Headers, &response)
	if err != nil {
		return nil, err
	}
	if response.PromptFeedback != nil && response.PromptFeedback.BlockReason != "" {
		return nil, goaierrors.ErrContentFiltered
	}
	var audio *geminiInlineData
	for _, candidate := range response.Candidates {
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				if part.InlineData != nil {
					audio = part.InlineData
					break
				}
			}
		}
		if audio != nil {
			break
		}
	}
	if audio == nil {
		return nil, fmt.Errorf("%w: no audio in Gemini response", goaierrors.ErrInvalidResponse)
	}
	pcm, err := base64.StdEncoding.DecodeString(audio.Data)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)
	}
	rate := 24000
	_, params, err := mime.ParseMediaType(audio.MimeType)
	if err == nil && params["rate"] != "" {
		rate, err = strconv.Atoi(params["rate"])
		if err != nil || rate <= 0 {
			return nil, fmt.Errorf("%w: invalid audio sample rate", goaierrors.ErrInvalidResponse)
		}
	}
	result := &model.SpeechResult{Audio: pcm, MimeType: audio.MimeType, Response: model.SpeechResponse{Model: m.id, Headers: headers}}
	if opts.Speed != nil {
		result.Warnings = append(result.Warnings, stream.UnsupportedWarning("speed", "Gemini TTS does not support speed."))
	}
	if opts.OutputFormat != "pcm" {
		if opts.OutputFormat != "" && opts.OutputFormat != "wav" {
			result.Warnings = append(result.Warnings, stream.UnsupportedWarning("outputFormat", "Using wav."))
		}
		wav := make([]byte, 44+len(pcm))
		copy(wav, "RIFF")
		binary.LittleEndian.PutUint32(wav[4:], uint32(36+len(pcm)))
		copy(wav[8:], "WAVEfmt ")
		binary.LittleEndian.PutUint32(wav[16:], 16)
		binary.LittleEndian.PutUint16(wav[20:], 1)
		binary.LittleEndian.PutUint16(wav[22:], 1)
		binary.LittleEndian.PutUint32(wav[24:], uint32(rate))
		binary.LittleEndian.PutUint32(wav[28:], uint32(rate*2))
		binary.LittleEndian.PutUint16(wav[32:], 2)
		binary.LittleEndian.PutUint16(wav[34:], 16)
		copy(wav[36:], "data")
		binary.LittleEndian.PutUint32(wav[40:], uint32(len(pcm)))
		copy(wav[44:], pcm)
		result.Audio = wav
		result.MimeType = "audio/wav"
	}
	return result, nil
}
