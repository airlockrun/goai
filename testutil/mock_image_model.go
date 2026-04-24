package testutil

import (
	"context"

	"github.com/airlockrun/goai/model"
)

// MockImageModel is a mock implementation of ImageModel for testing.
type MockImageModel struct {
	id               string
	provider         string
	maxImagesPerCall int
	DoGenerateFunc   func(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error)
	DoGenerateCalls  []model.ImageCallOptions
	GenerateResponse *model.ImageResult
}

// MockImageModelOptions configures the mock image model.
type MockImageModelOptions struct {
	ID               string
	Provider         string
	MaxImagesPerCall int
	DoGenerateFunc   func(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error)
	GenerateResponse *model.ImageResult
}

// NewMockImageModel creates a new mock image model.
func NewMockImageModel(opts MockImageModelOptions) *MockImageModel {
	m := &MockImageModel{
		id:               opts.ID,
		provider:         opts.Provider,
		maxImagesPerCall: opts.MaxImagesPerCall,
		DoGenerateFunc:   opts.DoGenerateFunc,
		GenerateResponse: opts.GenerateResponse,
	}
	if m.id == "" {
		m.id = "mock-model-id"
	}
	if m.provider == "" {
		m.provider = "mock-provider"
	}
	if m.maxImagesPerCall == 0 {
		m.maxImagesPerCall = 1
	}
	return m
}

func (m *MockImageModel) ID() string            { return m.id }
func (m *MockImageModel) Provider() string      { return m.provider }
func (m *MockImageModel) MaxImagesPerCall() int { return m.maxImagesPerCall }

func (m *MockImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	m.DoGenerateCalls = append(m.DoGenerateCalls, opts)

	if m.DoGenerateFunc != nil {
		return m.DoGenerateFunc(ctx, opts)
	}

	if m.GenerateResponse != nil {
		return m.GenerateResponse, nil
	}

	return &model.ImageResult{
		Images: []model.GeneratedImage{
			{
				Base64:   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
				MimeType: "image/png",
			},
		},
	}, nil
}
