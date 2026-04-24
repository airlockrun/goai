package goai

import (
	"context"
	"fmt"
	"io"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// TranscribeInput contains the input for transcription.
type TranscribeInput struct {
	// Model is the transcription model to use.
	Model model.TranscriptionModel

	// Audio is the audio data to transcribe.
	Audio []byte

	// AudioReader provides streaming access to audio data (alternative to Audio).
	AudioReader io.Reader

	// MimeType is the MIME type of the audio (e.g., "audio/wav", "audio/mp3").
	MimeType string

	// Filename is the filename of the audio (used for format detection).
	Filename string

	// Language is the language code of the audio (e.g., "en", "es").
	// If not provided, the model will attempt to detect the language.
	Language string

	// Prompt is an optional hint for the model (can improve accuracy).
	Prompt string

	// AbortSignal allows cancellation.
	AbortSignal context.Context

	// Headers are additional HTTP headers.
	Headers map[string]string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any
}

// TranscriptionResult contains the result of transcription.
type TranscriptionResult struct {
	// Text is the transcribed text.
	Text string

	// Segments contains detailed segment information (if available).
	Segments []TranscriptionSegment

	// Language is the detected language code.
	Language string

	// Duration is the duration of the audio in seconds (if available).
	Duration *float64

	// Warnings contains any warnings from the transcription process.
	Warnings []stream.Warning

	// Usage contains usage information (if available).
	Usage *TranscriptionUsage

	// Response contains response metadata.
	Response TranscriptionResponseMeta
}

// TranscriptionSegment represents a segment of transcribed audio.
type TranscriptionSegment struct {
	// ID is the segment identifier.
	ID int

	// Text is the transcribed text for this segment.
	Text string

	// Start is the start time in seconds.
	Start float64

	// End is the end time in seconds.
	End float64

	// Confidence is the confidence score (0 to 1).
	Confidence float64

	// Words contains word-level information (if available).
	Words []TranscriptionWord
}

// TranscriptionWord represents a single word in the transcription.
type TranscriptionWord struct {
	// Word is the transcribed word.
	Word string

	// Start is the start time in seconds.
	Start float64

	// End is the end time in seconds.
	End float64

	// Confidence is the confidence score (0 to 1).
	Confidence float64
}

// TranscriptionUsage contains usage information for transcription.
type TranscriptionUsage struct {
	// DurationSeconds is the duration of audio processed in seconds.
	DurationSeconds float64
}

// TranscriptionResponseMeta contains response metadata.
type TranscriptionResponseMeta struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for transcription.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}

// Transcribe transcribes audio to text using a transcription model.
func Transcribe(ctx context.Context, input TranscribeInput) (*TranscriptionResult, error) {
	if input.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if input.Audio == nil && input.AudioReader == nil {
		return nil, fmt.Errorf("audio or audioReader is required")
	}

	// Call the model
	modelResult, err := input.Model.Transcribe(ctx, model.TranscribeCallOptions{
		Audio:           input.Audio,
		AudioReader:     input.AudioReader,
		MimeType:        input.MimeType,
		Filename:        input.Filename,
		Language:        input.Language,
		Prompt:          input.Prompt,
		ProviderOptions: input.ProviderOptions,
		Headers:         input.Headers,
	})
	if err != nil {
		return nil, err
	}

	// Convert model result to goai result
	segments := make([]TranscriptionSegment, len(modelResult.Segments))
	for i, seg := range modelResult.Segments {
		words := make([]TranscriptionWord, len(seg.Words))
		for j, w := range seg.Words {
			words[j] = TranscriptionWord{
				Word:       w.Word,
				Start:      w.Start,
				End:        w.End,
				Confidence: w.Confidence,
			}
		}
		segments[i] = TranscriptionSegment{
			ID:         seg.ID,
			Text:       seg.Text,
			Start:      seg.Start,
			End:        seg.End,
			Confidence: seg.Confidence,
			Words:      words,
		}
	}

	var usage *TranscriptionUsage
	if modelResult.Usage != nil {
		usage = &TranscriptionUsage{
			DurationSeconds: modelResult.Usage.DurationSeconds,
		}
	}

	return &TranscriptionResult{
		Text:     modelResult.Text,
		Segments: segments,
		Language: modelResult.Language,
		Duration: modelResult.Duration,
		Warnings: modelResult.Warnings,
		Usage:    usage,
		Response: TranscriptionResponseMeta{
			ID:      modelResult.Response.ID,
			Model:   modelResult.Response.Model,
			Headers: modelResult.Response.Headers,
		},
	}, nil
}
