package testutil

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// MockTranscriptionModel is a mock implementation of TranscriptionModel for testing.
type MockTranscriptionModel struct {
	id                 string
	provider           string
	DoTranscribeFunc   func(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error)
	DoTranscribeCalls  []model.TranscribeCallOptions
	TranscribeResponse *model.TranscriptionResult
}

// MockTranscriptionModelOptions configures the mock transcription model.
type MockTranscriptionModelOptions struct {
	ID                 string
	Provider           string
	DoTranscribeFunc   func(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error)
	TranscribeResponse *model.TranscriptionResult
}

// NewMockTranscriptionModel creates a new mock transcription model.
func NewMockTranscriptionModel(opts MockTranscriptionModelOptions) *MockTranscriptionModel {
	m := &MockTranscriptionModel{
		id:                 opts.ID,
		provider:           opts.Provider,
		DoTranscribeFunc:   opts.DoTranscribeFunc,
		TranscribeResponse: opts.TranscribeResponse,
	}
	if m.id == "" {
		m.id = "mock-model-id"
	}
	if m.provider == "" {
		m.provider = "mock-provider"
	}
	return m
}

func (m *MockTranscriptionModel) ID() string       { return m.id }
func (m *MockTranscriptionModel) Provider() string { return m.provider }

func (m *MockTranscriptionModel) Transcribe(ctx context.Context, opts model.TranscribeCallOptions) (*model.TranscriptionResult, error) {
	m.DoTranscribeCalls = append(m.DoTranscribeCalls, opts)

	if m.DoTranscribeFunc != nil {
		return m.DoTranscribeFunc(ctx, opts)
	}

	if m.TranscribeResponse != nil {
		return m.TranscribeResponse, nil
	}

	duration := 4.0
	return &model.TranscriptionResult{
		Text:     "This is a sample transcript.",
		Language: "en",
		Duration: &duration,
		Segments: []model.TranscriptionSegment{
			{ID: 0, Text: "This is a", Start: 0, End: 2.5},
			{ID: 1, Text: "sample transcript.", Start: 2.5, End: 4.0},
		},
	}, nil
}
