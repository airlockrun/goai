package vertex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/google"
	"github.com/airlockrun/goai/stream"
)

type VertexSpeechModel struct {
	id       string
	provider *Provider
}

func (m *VertexSpeechModel) ID() string       { return m.id }
func (m *VertexSpeechModel) Provider() string { return "vertex" }
func (m *VertexSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	if !strings.HasPrefix(m.id, "chirp-") {
		headers := map[string]string{}
		for k, v := range m.provider.opts.Headers {
			headers[k] = v
		}
		headers["Authorization"] = "Bearer " + m.provider.opts.AccessToken
		return google.New(google.Options{BaseURL: m.provider.baseURL() + "/publishers/google", Headers: headers}).SpeechModel(m.id).Generate(ctx, opts)
	}
	voice := opts.Voice
	if voice == "" {
		voice = "Kore"
	}
	language := "en-US"
	if value, ok := opts.ProviderOptions["language"].(string); ok {
		language = value
	}
	if index := strings.Index(voice, "Chirp3-HD"); index >= 0 {
		if index > 0 {
			language = strings.TrimSuffix(voice[:index], "-")
		}
	} else {
		voice = language + "-Chirp3-HD-" + voice
	}
	config := map[string]any{"audioEncoding": "LINEAR16"}
	if opts.Speed != nil {
		config["speakingRate"] = *opts.Speed
	}
	request := map[string]any{"input": map[string]any{"text": opts.Text}, "voice": map[string]any{"languageCode": language, "name": voice}, "audioConfig": config}
	url := "https://texttospeech.googleapis.com/v1/text:synthesize"
	if m.provider.opts.BaseURL != "" {
		url = strings.TrimRight(m.provider.opts.BaseURL, "/") + "/v1/text:synthesize"
	}
	var response struct {
		Audio string `json:"audioContent"`
	}
	headers, err := m.provider.postAudio(ctx, url, request, opts.Headers, &response)
	if err != nil {
		return nil, err
	}
	audio, err := base64.StdEncoding.DecodeString(response.Audio)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)
	}
	result := &model.SpeechResult{Audio: audio, MimeType: "audio/wav", Response: model.SpeechResponse{Model: m.id, Headers: headers}}
	if opts.OutputFormat != "" && opts.OutputFormat != "wav" {
		result.Warnings = append(result.Warnings, stream.UnsupportedWarning("outputFormat", "Using wav."))
	}
	return result, nil
}

type VertexTranscriptionModel struct {
	id       string
	provider *Provider
}

func (m *VertexTranscriptionModel) ID() string       { return m.id }
func (m *VertexTranscriptionModel) Provider() string { return "vertex" }
func (m *VertexTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
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
	content := base64.StdEncoding.EncodeToString(audio)
	region := m.provider.opts.Location
	if value, ok := opts.ProviderOptions["region"].(string); ok {
		region = value
	}
	languages := any([]string{"auto"})
	if opts.Language != "" {
		languages = []string{opts.Language}
	}
	if value, ok := opts.ProviderOptions["languageCodes"]; ok {
		languages = value
	}
	features := map[string]any{"enableWordTimeOffsets": true, "enableAutomaticPunctuation": true}
	for key := range features {
		if value, ok := opts.ProviderOptions[key]; ok {
			features[key] = value
		}
	}
	request := map[string]any{"content": content, "config": map[string]any{"model": m.id, "languageCodes": languages, "autoDecodingConfig": map[string]any{}, "features": features}}
	host := region + "-speech.googleapis.com"
	if region == "global" {
		host = "speech.googleapis.com"
	}
	base := "https://" + host
	if m.provider.opts.BaseURL != "" {
		base = strings.TrimRight(m.provider.opts.BaseURL, "/")
	}
	url := fmt.Sprintf("%s/v2/projects/%s/locations/%s/recognizers/_:recognize", base, m.provider.opts.ProjectID, region)
	if strings.HasPrefix(m.id, "gemini-") {
		config := map[string]any{}
		for _, key := range []string{"languageCodes", "customVocabulary", "wordTimestamp", "diarization", "mode"} {
			if value, ok := opts.ProviderOptions[key]; ok {
				config[key] = value
			}
		}
		if opts.Language != "" {
			if _, ok := config["languageCodes"]; !ok {
				config["languageCodes"] = []string{opts.Language}
			}
		}
		request = map[string]any{"contents": []any{map[string]any{"role": "user", "parts": []any{map[string]any{"inlineData": map[string]any{"mimeType": opts.MimeType, "data": content}}}}}}
		if len(config) > 0 {
			request["generationConfig"] = map[string]any{"audioTranscriptionConfig": config}
		}
		url = m.provider.baseURL() + "/publishers/google/models/" + m.id + ":generateContent"
	}
	type word struct {
		Text  string `json:"word"`
		Start string `json:"startOffset"`
		End   string `json:"endOffset"`
	}
	var response struct {
		Results []struct {
			Language     string `json:"languageCode"`
			Alternatives []struct {
				Text  string `json:"transcript"`
				Words []word `json:"words"`
			} `json:"alternatives"`
		} `json:"results"`
		Metadata struct {
			Duration string `json:"totalBilledDuration"`
		} `json:"metadata"`
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text               string `json:"text"`
					AudioTranscription *struct {
						Text     string `json:"text"`
						Language string `json:"languageCode"`
						Words    []word `json:"words"`
					} `json:"audioTranscription"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		PromptFeedback struct {
			BlockReason string `json:"blockReason"`
		} `json:"promptFeedback"`
	}
	headers, err := m.provider.postAudio(ctx, url, request, opts.Headers, &response)
	if err != nil {
		return nil, err
	}
	if response.PromptFeedback.BlockReason != "" {
		return nil, goaierrors.ErrContentFiltered
	}
	result := &model.TranscriptionResult{Response: model.TranscriptionResponse{Model: m.id, Headers: headers}}
	var words []word
	for _, part := range response.Results {
		if len(part.Alternatives) > 0 {
			if result.Text != "" {
				result.Text += " "
			}
			result.Text += part.Alternatives[0].Text
			words = append(words, part.Alternatives[0].Words...)
			if result.Language == "" {
				result.Language = strings.Split(part.Language, "-")[0]
			}
		}
	}
	if len(response.Candidates) > 0 {
		var transcriptionText string
		for _, part := range response.Candidates[0].Content.Parts {
			result.Text += part.Text
			if part.AudioTranscription != nil {
				transcriptionText += part.AudioTranscription.Text
				words = append(words, part.AudioTranscription.Words...)
				if result.Language == "" {
					result.Language = part.AudioTranscription.Language
				}
			}
		}
		if result.Text == "" {
			result.Text = transcriptionText
		}
	}
	for _, word := range words {
		start, e1 := strconv.ParseFloat(strings.TrimSuffix(word.Start, "s"), 64)
		end, e2 := strconv.ParseFloat(strings.TrimSuffix(word.End, "s"), 64)
		if e1 == nil && e2 == nil {
			result.Segments = append(result.Segments, model.TranscriptionSegment{ID: len(result.Segments), Text: word.Text, Start: start, End: end})
		}
	}
	if seconds, err := strconv.ParseFloat(strings.TrimSuffix(response.Metadata.Duration, "s"), 64); err == nil {
		result.Duration = &seconds
	}
	if opts.Prompt != "" {
		result.Warnings = append(result.Warnings, stream.UnsupportedWarning("prompt", "Use customVocabulary for Gemini transcription."))
	}
	return result, nil
}

func (p *Provider) postAudio(ctx context.Context, url string, body any, headers map[string]string, result any) (map[string]string, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range p.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Authorization", "Bearer "+p.opts.AccessToken)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: "Vertex audio request failed", URL: url, Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true})
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: string(raw), URL: url, StatusCode: resp.StatusCode, ResponseBody: string(raw)})
	}
	if err := json.Unmarshal(raw, result); err != nil {
		return nil, fmt.Errorf("%w: %v", goaierrors.ErrInvalidResponse, err)
	}
	responseHeaders := map[string]string{}
	for key := range resp.Header {
		responseHeaders[key] = resp.Header.Get(key)
	}
	return responseHeaders, nil
}
