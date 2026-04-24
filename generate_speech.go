package goai

import (
	"context"
	"fmt"
	"io"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// SpeechInput contains the input for speech generation.
type SpeechInput struct {
	// Model is the speech model to use.
	Model model.SpeechModel

	// Text is the text to convert to speech.
	Text string

	// Voice is the voice to use for generation.
	Voice string

	// OutputFormat is the desired output format (e.g., "mp3", "wav", "opus").
	// Default depends on the provider.
	OutputFormat string

	// Speed is the speed of the generated audio (0.25 to 4.0, 1.0 is normal).
	Speed *float64

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any
}

// SpeechResult contains the result of speech generation.
type SpeechResult struct {
	// Audio is the generated audio data.
	Audio []byte

	// AudioReader provides streaming access to the audio data.
	AudioReader io.Reader

	// MimeType is the MIME type of the audio (e.g., "audio/mpeg").
	MimeType string

	// Duration is the duration of the audio in seconds (if available).
	Duration *float64

	// Warnings contains any warnings from the generation process.
	Warnings []stream.Warning

	// Usage contains usage information (if available).
	Usage *SpeechUsage

	// Response contains response metadata.
	Response SpeechResponseMeta
}

// SpeechUsage contains usage information for speech generation.
type SpeechUsage struct {
	// Characters is the number of characters processed.
	Characters int

	// Seconds is the duration of generated audio in seconds.
	Seconds float64
}

// SpeechResponseMeta contains response metadata.
type SpeechResponseMeta struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}

// GenerateSpeech generates speech from text using a speech model.
func GenerateSpeech(ctx context.Context, input SpeechInput) (*SpeechResult, error) {
	if input.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if input.Text == "" {
		return nil, fmt.Errorf("text is required")
	}

	// Call the model
	modelResult, err := input.Model.Generate(ctx, model.SpeechCallOptions{
		Text:            input.Text,
		Voice:           input.Voice,
		OutputFormat:    input.OutputFormat,
		Speed:           input.Speed,
		ProviderOptions: input.ProviderOptions,
		Headers:         input.Headers,
	})
	if err != nil {
		return nil, err
	}

	// Convert model result to goai result
	var usage *SpeechUsage
	if modelResult.Usage != nil {
		usage = &SpeechUsage{
			Characters: modelResult.Usage.Characters,
			Seconds:    modelResult.Usage.Seconds,
		}
	}

	return &SpeechResult{
		Audio:       modelResult.Audio,
		AudioReader: modelResult.AudioReader,
		MimeType:    modelResult.MimeType,
		Duration:    modelResult.Duration,
		Warnings:    modelResult.Warnings,
		Usage:       usage,
		Response: SpeechResponseMeta{
			ID:      modelResult.Response.ID,
			Model:   modelResult.Response.Model,
			Headers: modelResult.Response.Headers,
		},
	}, nil
}
