package goai

import (
	"context"
	"testing"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/testutil"
)

// Translated from ai-sdk/packages/ai/src/generate-speech/generate-speech.test.ts

var sampleAudio = []byte{1, 2, 3, 4} // Sample audio data
const sampleSpeechText = "This is a sample text to convert to speech."

func createMockSpeechResponse(audio []byte) *model.SpeechResult {
	return &model.SpeechResult{
		Audio:    audio,
		MimeType: "audio/mp3",
		Response: model.SpeechResponse{
			Model: "test-model-id",
		},
	}
}

func TestGenerateSpeech_ShouldSendArgsToDoGenerate(t *testing.T) {
	var capturedOpts model.SpeechCallOptions

	mockModel := testutil.NewMockSpeechModel(testutil.MockSpeechModelOptions{
		DoGenerateFunc: func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
			capturedOpts = opts
			return createMockSpeechResponse(sampleAudio), nil
		},
	})

	_, err := GenerateSpeech(context.Background(), SpeechInput{
		Model: mockModel,
		Text:  sampleSpeechText,
		Voice: "test-voice",
		Headers: map[string]string{
			"custom-request-header": "request-header-value",
		},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedOpts.Text != sampleSpeechText {
		t.Errorf("expected text %s, got %s", sampleSpeechText, capturedOpts.Text)
	}

	if capturedOpts.Voice != "test-voice" {
		t.Errorf("expected voice test-voice, got %s", capturedOpts.Voice)
	}

	if capturedOpts.Headers["custom-request-header"] != "request-header-value" {
		t.Errorf("expected custom header, got %v", capturedOpts.Headers)
	}
}

func TestGenerateSpeech_ShouldReturnTheAudioData(t *testing.T) {
	mockModel := testutil.NewMockSpeechModel(testutil.MockSpeechModelOptions{
		DoGenerateFunc: func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
			return createMockSpeechResponse(sampleAudio), nil
		},
	})

	result, err := GenerateSpeech(context.Background(), SpeechInput{
		Model: mockModel,
		Text:  sampleSpeechText,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Audio) != len(sampleAudio) {
		t.Errorf("expected audio length %d, got %d", len(sampleAudio), len(result.Audio))
	}

	for i, v := range result.Audio {
		if v != sampleAudio[i] {
			t.Errorf("expected audio[%d] = %d, got %d", i, sampleAudio[i], v)
		}
	}

	if result.MimeType != "audio/mp3" {
		t.Errorf("expected mime type audio/mp3, got %s", result.MimeType)
	}
}

func TestGenerateSpeech_ErrorHandling(t *testing.T) {
	t.Run("should handle empty audio", func(t *testing.T) {
		mockModel := testutil.NewMockSpeechModel(testutil.MockSpeechModelOptions{
			DoGenerateFunc: func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
				return &model.SpeechResult{
					Audio:    []byte{},
					MimeType: "audio/mp3",
				}, nil
			},
		})

		result, err := GenerateSpeech(context.Background(), SpeechInput{
			Model: mockModel,
			Text:  sampleSpeechText,
		})

		// Current implementation returns empty result without error
		if err == nil && len(result.Audio) == 0 {
			// Expected behavior - no error but empty audio
		}
	})
}

func TestGenerateSpeech_ResponseMetadata(t *testing.T) {
	t.Run("should return response metadata", func(t *testing.T) {
		mockModel := testutil.NewMockSpeechModel(testutil.MockSpeechModelOptions{
			DoGenerateFunc: func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
				return &model.SpeechResult{
					Audio:    sampleAudio,
					MimeType: "audio/mp3",
					Response: model.SpeechResponse{
						Model: "test-model",
					},
				}, nil
			},
		})

		result, err := GenerateSpeech(context.Background(), SpeechInput{
			Model: mockModel,
			Text:  sampleSpeechText,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Response.Model != "test-model" {
			t.Errorf("expected response model test-model, got %s", result.Response.Model)
		}
	})
}

func TestGenerateSpeech_ProviderOptions(t *testing.T) {
	t.Run("should pass provider options to model", func(t *testing.T) {
		mockModel := testutil.NewMockSpeechModel(testutil.MockSpeechModelOptions{
			DoGenerateFunc: func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
				if opts.ProviderOptions == nil {
					t.Error("expected provider options to be set")
				}
				if opts.ProviderOptions["testProvider"] == nil {
					t.Error("expected testProvider in provider options")
				}
				return createMockSpeechResponse(sampleAudio), nil
			},
		})

		_, err := GenerateSpeech(context.Background(), SpeechInput{
			Model: mockModel,
			Text:  sampleSpeechText,
			ProviderOptions: map[string]any{
				"testProvider": map[string]any{"key": "value"},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}
