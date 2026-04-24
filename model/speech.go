package model

import (
	"context"
	"io"

	"github.com/airlockrun/goai/stream"
)

// SpeechModel is the interface for text-to-speech models.
type SpeechModel interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// Generate generates speech from text.
	Generate(ctx context.Context, opts SpeechCallOptions) (*SpeechResult, error)
}

// SpeechCallOptions contains the options for speech generation.
type SpeechCallOptions struct {
	// Text is the text to convert to speech.
	Text string

	// Voice is the voice to use for generation.
	Voice string

	// OutputFormat is the desired output format (e.g., "mp3", "wav", "opus").
	OutputFormat string

	// Speed is the speed of the generated audio (0.25 to 4.0, 1.0 is normal).
	Speed *float64

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Headers are additional HTTP headers.
	Headers map[string]string
}

// SpeechResult contains the result of a speech generation call.
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

	// Response contains provider-specific response data.
	Response SpeechResponse
}

// SpeechUsage contains usage information for speech generation.
type SpeechUsage struct {
	// Characters is the number of characters processed.
	Characters int

	// Seconds is the duration of generated audio in seconds.
	Seconds float64
}

// SpeechResponse contains provider-specific response metadata.
type SpeechResponse struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for generation.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}
