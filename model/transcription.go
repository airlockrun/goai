package model

import (
	"context"
	"io"

	"github.com/airlockrun/goai/stream"
)

// TranscriptionModel is the interface for speech-to-text models.
type TranscriptionModel interface {
	// ID returns the model identifier.
	ID() string

	// Provider returns the provider identifier.
	Provider() string

	// Transcribe transcribes audio to text.
	Transcribe(ctx context.Context, opts TranscribeCallOptions) (*TranscriptionResult, error)
}

// TranscribeCallOptions contains the options for transcription.
type TranscribeCallOptions struct {
	// Audio is the audio data to transcribe.
	Audio []byte

	// AudioReader provides streaming access to audio data (alternative to Audio).
	AudioReader io.Reader

	// AudioURL is a URL to the audio file (alternative to Audio/AudioReader).
	AudioURL string

	// MimeType is the MIME type of the audio (e.g., "audio/wav", "audio/mp3").
	MimeType string

	// Filename is the filename of the audio (used for format detection).
	Filename string

	// Language is the language code of the audio (e.g., "en", "es").
	// If not provided, the model will attempt to detect the language.
	Language string

	// Prompt is an optional hint for the model (can improve accuracy).
	Prompt string

	// ProviderOptions are provider-specific options.
	ProviderOptions map[string]any

	// Headers are additional HTTP headers.
	Headers map[string]string
}

// TranscriptionResult contains the result of a transcription call.
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

	// Response contains provider-specific response data.
	Response TranscriptionResponse
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

// TranscriptionResponse contains provider-specific response metadata.
type TranscriptionResponse struct {
	// ID is the response identifier.
	ID string

	// Model is the model used for transcription.
	Model string

	// Headers contains response headers.
	Headers map[string]string
}
