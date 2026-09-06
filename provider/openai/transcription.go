package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

// OpenAITranscriptionModel implements the TranscriptionModel interface for OpenAI.
type OpenAITranscriptionModel struct {
	id       string
	provider *Provider
}

// ID returns the model identifier.
func (m *OpenAITranscriptionModel) ID() string {
	return m.id
}

// Provider returns "openai".
func (m *OpenAITranscriptionModel) Provider() string {
	return "openai"
}

// Transcribe transcribes audio to text.
func (m *OpenAITranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	result, err := m.TranscribeDetailed(ctx, opts)
	if err != nil {
		return nil, err
	}
	if len(result.DiarizedSegments) > 0 {
		result.Warnings = append(result.Warnings, stream.UnsupportedWarning("speaker", "use TranscribeDetailed to access speaker labels and string segment IDs"))
	}
	return &result.TranscriptionResult, nil
}

// DiarizedSegment preserves the provider's segment identity and speaker label.
type DiarizedSegment struct {
	ID      string  `json:"id"`
	Speaker string  `json:"speaker"`
	Text    string  `json:"text"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
}

// DetailedTranscriptionResult includes diarization data not represented by model.TranscriptionResult.
type DetailedTranscriptionResult struct {
	model.TranscriptionResult
	DiarizedSegments []DiarizedSegment
}

// TranscriptionOptions controls the audio transcription endpoint.
type TranscriptionOptions struct {
	ResponseFormat         string   `json:"responseFormat,omitempty"`
	ChunkingStrategy       any      `json:"chunkingStrategy,omitempty"`
	KnownSpeakerNames      []string `json:"knownSpeakerNames,omitempty"`
	KnownSpeakerReferences []string `json:"knownSpeakerReferences,omitempty"`
	Temperature            *float64 `json:"temperature,omitempty"`
	TimestampGranularities []string `json:"timestampGranularities,omitempty"`
}

// TranscribeDetailed transcribes audio and preserves diarized speaker segments.
func (m *OpenAITranscriptionModel) TranscribeDetailed(ctx context.Context, opts model.TranscribeCallOptions) (*DetailedTranscriptionResult, error) {
	providerOpts, err := provider.ParseProviderOptions[TranscriptionOptions](opts.ProviderOptions)
	if err != nil {
		return nil, err
	}
	// Read audio data if provided as reader
	var audioData []byte
	if opts.Audio != nil {
		audioData = opts.Audio
	} else if opts.AudioReader != nil {
		var err error
		audioData, err = io.ReadAll(opts.AudioReader)
		if err != nil {
			return nil, fmt.Errorf("failed to read audio data: %w", err)
		}
	} else {
		return nil, fmt.Errorf("audio data is required")
	}

	// Determine filename
	filename := opts.Filename
	if filename == "" {
		// Default filename based on MIME type
		switch opts.MimeType {
		case "audio/wav", "audio/wave":
			filename = "audio.wav"
		case "audio/mp3", "audio/mpeg":
			filename = "audio.mp3"
		case "audio/m4a":
			filename = "audio.m4a"
		case "audio/webm":
			filename = "audio.webm"
		case "audio/ogg":
			filename = "audio.ogg"
		case "audio/flac":
			filename = "audio.flac"
		default:
			filename = "audio.mp3"
		}
	}

	// Build multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(audioData); err != nil {
		return nil, fmt.Errorf("failed to write audio data: %w", err)
	}

	// Add model
	if err := writer.WriteField("model", m.id); err != nil {
		return nil, fmt.Errorf("failed to write model field: %w", err)
	}

	// Add language if specified
	if opts.Language != "" {
		if err := writer.WriteField("language", opts.Language); err != nil {
			return nil, fmt.Errorf("failed to write language field: %w", err)
		}
	}

	// Add prompt if specified
	if opts.Prompt != "" {
		if err := writer.WriteField("prompt", opts.Prompt); err != nil {
			return nil, fmt.Errorf("failed to write prompt field: %w", err)
		}
	}

	// Prefer verbose_json for segments/words, but gpt-4o-transcribe and
	// gpt-4o-mini-transcribe only accept `json` or `text` — match ai-sdk's
	// exclusion list. timestamp_granularities is only valid with verbose_json.
	responseFormat := "verbose_json"
	if strings.HasPrefix(m.id, "gpt-4o-transcribe") || strings.HasPrefix(m.id, "gpt-4o-mini-transcribe") {
		responseFormat = "json"
	}
	if strings.HasPrefix(m.id, "gpt-4o-transcribe-diarize") {
		responseFormat = "diarized_json"
		if providerOpts.ChunkingStrategy == nil {
			providerOpts.ChunkingStrategy = "auto"
		}
	}
	if providerOpts.ResponseFormat != "" {
		responseFormat = providerOpts.ResponseFormat
	}
	if responseFormat != "json" && responseFormat != "verbose_json" && responseFormat != "diarized_json" {
		return nil, fmt.Errorf("unsupported transcription response format: %s", responseFormat)
	}
	if providerOpts.ChunkingStrategy != nil {
		value, ok := providerOpts.ChunkingStrategy.(string)
		if !ok {
			encoded, err := json.Marshal(providerOpts.ChunkingStrategy)
			if err != nil {
				return nil, err
			}
			value = string(encoded)
		}
		if err := writer.WriteField("chunking_strategy", value); err != nil {
			return nil, err
		}
	}
	for key, values := range map[string][]string{"known_speaker_names[]": providerOpts.KnownSpeakerNames, "known_speaker_references[]": providerOpts.KnownSpeakerReferences} {
		for _, value := range values {
			if err := writer.WriteField(key, value); err != nil {
				return nil, err
			}
		}
	}
	if providerOpts.Temperature != nil {
		if err := writer.WriteField("temperature", strconv.FormatFloat(*providerOpts.Temperature, 'f', -1, 64)); err != nil {
			return nil, err
		}
	}
	if err := writer.WriteField("response_format", responseFormat); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %w", err)
	}
	if responseFormat == "verbose_json" {
		granularities := providerOpts.TimestampGranularities
		if granularities == nil {
			granularities = []string{"word", "segment"}
		}
		for _, value := range granularities {
			if err := writer.WriteField("timestamp_granularities[]", value); err != nil {
				return nil, err
			}
		}
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.provider.opts.BaseURL+"/audio/transcriptions", &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	httpReq.Header.Set("Authorization", "Bearer "+m.provider.opts.APIKey)
	if m.provider.opts.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", m.provider.opts.Organization)
	}
	if m.provider.opts.Project != "" {
		httpReq.Header.Set("OpenAI-Project", m.provider.opts.Project)
	}
	// Provider-level headers
	for k, v := range m.provider.opts.Headers {
		httpReq.Header.Set(k, v)
	}
	// Request-level headers (override provider headers)
	for k, v := range opts.Headers {
		httpReq.Header.Set(k, v)
	}

	// Execute request
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var transResp transcriptionResponse
	if err := json.Unmarshal(body, &transResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to model result
	segments := make([]model.TranscriptionSegment, len(transResp.Segments))
	var diarized []DiarizedSegment
	for i, seg := range transResp.Segments {
		id := i
		var numericID int
		if json.Unmarshal(seg.ID, &numericID) == nil {
			id = numericID
		}
		if seg.Speaker != "" || responseFormat == "diarized_json" {
			var stringID string
			if json.Unmarshal(seg.ID, &stringID) != nil {
				stringID = string(seg.ID)
			}
			diarized = append(diarized, DiarizedSegment{ID: stringID, Speaker: seg.Speaker, Text: seg.Text, Start: seg.Start, End: seg.End})
		}
		words := make([]model.TranscriptionWord, len(seg.Words))
		for j, w := range seg.Words {
			words[j] = model.TranscriptionWord{
				Word:  w.Word,
				Start: w.Start,
				End:   w.End,
			}
		}
		segments[i] = model.TranscriptionSegment{
			ID:    id,
			Text:  seg.Text,
			Start: seg.Start,
			End:   seg.End,
			Words: words,
		}
	}
	if transResp.Duration == 0 {
		for _, segment := range segments {
			if segment.End > transResp.Duration {
				transResp.Duration = segment.End
			}
		}
	}

	var duration *float64
	if transResp.Duration > 0 {
		duration = &transResp.Duration
	}

	return &DetailedTranscriptionResult{DiarizedSegments: diarized, TranscriptionResult: model.TranscriptionResult{
		Text:     transResp.Text,
		Segments: segments,
		Language: transResp.Language,
		Duration: duration,
		Usage: &model.TranscriptionUsage{
			DurationSeconds: transResp.Duration,
		},
		Response: model.TranscriptionResponse{
			Model: m.id,
		},
	}}, nil
}

// Response types

type transcriptionResponse struct {
	Task     string                 `json:"task"`
	Language string                 `json:"language"`
	Duration float64                `json:"duration"`
	Text     string                 `json:"text"`
	Words    []transcriptionWord    `json:"words,omitempty"`
	Segments []transcriptionSegment `json:"segments,omitempty"`
}

type transcriptionSegment struct {
	ID      json.RawMessage     `json:"id"`
	Speaker string              `json:"speaker,omitempty"`
	Start   float64             `json:"start"`
	End     float64             `json:"end"`
	Text    string              `json:"text"`
	Words   []transcriptionWord `json:"words,omitempty"`
}

type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}
