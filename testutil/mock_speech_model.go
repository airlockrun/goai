package testutil

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// MockSpeechModel is a mock implementation of SpeechModel for testing.
type MockSpeechModel struct {
	id               string
	provider         string
	DoGenerateFunc   func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error)
	DoGenerateCalls  []model.SpeechCallOptions
	GenerateResponse *model.SpeechResult
}

// MockSpeechModelOptions configures the mock speech model.
type MockSpeechModelOptions struct {
	ID               string
	Provider         string
	DoGenerateFunc   func(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error)
	GenerateResponse *model.SpeechResult
}

// NewMockSpeechModel creates a new mock speech model.
func NewMockSpeechModel(opts MockSpeechModelOptions) *MockSpeechModel {
	m := &MockSpeechModel{
		id:               opts.ID,
		provider:         opts.Provider,
		DoGenerateFunc:   opts.DoGenerateFunc,
		GenerateResponse: opts.GenerateResponse,
	}
	if m.id == "" {
		m.id = "mock-model-id"
	}
	if m.provider == "" {
		m.provider = "mock-provider"
	}
	return m
}

func (m *MockSpeechModel) ID() string       { return m.id }
func (m *MockSpeechModel) Provider() string { return m.provider }

func (m *MockSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	m.DoGenerateCalls = append(m.DoGenerateCalls, opts)

	if m.DoGenerateFunc != nil {
		return m.DoGenerateFunc(ctx, opts)
	}

	if m.GenerateResponse != nil {
		return m.GenerateResponse, nil
	}

	return &model.SpeechResult{
		Audio:    []byte{1, 2, 3, 4},
		MimeType: "audio/mp3",
	}, nil
}
