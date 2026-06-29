package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// chatAudioResult is the accumulated output of a streamed chat-audio call.
type chatAudioResult struct {
	audio      []byte
	transcript string
	text       string
}

// streamChatAudio issues a streaming /chat/completions call and accumulates the
// text, audio bytes, and audio transcript across SSE chunks. Audio output
// requires stream:true (the provider rejects non-streaming audio), so the
// adapters consume the stream rather than a single JSON response.
func (p *Provider) streamChatAudio(ctx context.Context, req chatRequest, headers map[string]string) (*chatAudioResult, error) {
	req.Stream = true
	req.StreamOptions = &chatStreamOptions{IncludeUsage: true}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.opts.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.opts.APIKey)
	if p.opts.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", p.opts.Organization)
	}
	if p.opts.Project != "" {
		httpReq.Header.Set("OpenAI-Project", p.opts.Project)
	}
	for k, v := range p.opts.Headers {
		httpReq.Header.Set(k, v)
	}
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(raw))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB: audio base64 lines are large

	var dataB64, transcript, text strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Error != nil {
			return nil, fmt.Errorf("OpenAI API error: %s", chunk.Error.Message)
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				text.WriteString(ch.Delta.Content)
			}
			if ch.Delta.Audio != nil {
				dataB64.WriteString(ch.Delta.Audio.Data)
				transcript.WriteString(ch.Delta.Audio.Transcript)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read stream: %w", err)
	}

	res := &chatAudioResult{transcript: transcript.String(), text: text.String()}
	if dataB64.Len() > 0 {
		audio, err := base64.StdEncoding.DecodeString(dataB64.String())
		if err != nil {
			return nil, fmt.Errorf("failed to decode audio: %w", err)
		}
		res.audio = audio
	}
	return res, nil
}

// chatSpeechModel adapts a chat-audio model to the SpeechModel interface by
// asking /chat/completions for spoken output (modalities:["text","audio"]).
type chatSpeechModel struct {
	id       string
	provider *Provider
}

func (m *chatSpeechModel) ID() string       { return m.id }
func (m *chatSpeechModel) Provider() string { return "openai" }

func (m *chatSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	msgs, err := convertToChatMessages([]message.Message{message.NewUserMessage(opts.Text)})
	if err != nil {
		return nil, err
	}

	voice := opts.Voice
	if voice == "" {
		voice = "alloy"
	}

	// Audio output requires stream:true, and in streaming mode OpenAI accepts
	// only pcm16 (raw 16-bit PCM, 24kHz mono). Other formats are rejected, so
	// the format is fixed; warn if the caller asked for something else.
	var warnings []stream.Warning
	const format = "pcm16"
	if of := opts.OutputFormat; of != "" && of != "pcm16" && of != "pcm" {
		warnings = append(warnings, stream.UnsupportedWarning("outputFormat",
			fmt.Sprintf("chat-audio streaming supports only pcm16; ignoring %q", of)))
	}
	if opts.Speed != nil {
		warnings = append(warnings, stream.UnsupportedWarning("speed",
			"speed is not supported by chat-audio models"))
	}

	req := chatRequest{
		Model:      m.id,
		Messages:   msgs,
		Modalities: []string{"text", "audio"},
		Audio:      &chatAudioConfig{Voice: voice, Format: format},
	}
	res, err := m.provider.streamChatAudio(ctx, req, opts.Headers)
	if err != nil {
		return nil, err
	}
	if len(res.audio) == 0 {
		return nil, fmt.Errorf("openai chat-audio model %q returned no audio", m.id)
	}

	// The stream yields raw pcm16; wrap it in a WAV container so callers get
	// directly-playable audio rather than headerless PCM.
	return &model.SpeechResult{
		Audio:    pcm16ToWAV(res.audio),
		MimeType: "audio/wav",
		Warnings: warnings,
		Usage:    &model.SpeechUsage{Characters: len(opts.Text)},
		Response: model.SpeechResponse{Model: m.id},
	}, nil
}

// pcm16ToWAV wraps raw little-endian 16-bit PCM in a WAV container. gpt-audio
// streaming output is 24kHz, mono, 16-bit.
func pcm16ToWAV(pcm []byte) []byte {
	const sampleRate, channels, bits = 24000, 1, 16
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+len(pcm)))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate*channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(bits))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}

// chatTranscriptionModel adapts a chat-audio model to the TranscriptionModel
// interface: it sends the audio as an input_audio part and requests text-only
// output (modalities:["text"]), returning the model's text as the transcript.
type chatTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *chatTranscriptionModel) ID() string       { return m.id }
func (m *chatTranscriptionModel) Provider() string { return "openai" }

func (m *chatTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	var audioData []byte
	switch {
	case opts.Audio != nil:
		audioData = opts.Audio
	case opts.AudioReader != nil:
		var err error
		audioData, err = io.ReadAll(opts.AudioReader)
		if err != nil {
			return nil, fmt.Errorf("failed to read audio data: %w", err)
		}
	default:
		return nil, fmt.Errorf("audio data is required")
	}

	mimeType := opts.MimeType
	if mimeType == "" {
		mimeType = "audio/wav"
	}
	// The audio rides as an input_audio part with modalities:["text"] requested.
	// A chat-audio model is conversational, so it tends to REPLY to the audio
	// rather than transcribe it; an instruction to make it transcribe (e.g.
	// "echo the user message verbatim") is application policy and is passed in via
	// opts.Prompt as a system message — goai supplies none of its own.
	var src []message.Message
	if opts.Prompt != "" {
		src = append(src, message.NewSystemMessage(opts.Prompt))
	}
	src = append(src, message.NewUserMessageWithParts(message.FilePart{
		Data:     message.FileDataBytes{Data: base64.StdEncoding.EncodeToString(audioData)},
		MimeType: mimeType,
	}))
	msgs, err := convertToChatMessages(src)
	if err != nil {
		return nil, err
	}

	req := chatRequest{
		Model:      m.id,
		Messages:   msgs,
		Modalities: []string{"text"},
	}
	res, err := m.provider.streamChatAudio(ctx, req, opts.Headers)
	if err != nil {
		return nil, err
	}
	text := res.text
	if text == "" {
		text = res.transcript
	}
	return &model.TranscriptionResult{
		Text:     text,
		Response: model.TranscriptionResponse{Model: m.id},
	}, nil
}
