package middleware

import (
	"context"
	"errors"
	"testing"

	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
)

func TestWrapModel(t *testing.T) {
	t.Run("passes through with no middleware", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Hello", testutil.MockUsage(10, 5)),
		})

		wrapped := WrapModel(model)

		if wrapped.ID() != model.ID() {
			t.Errorf("expected ID %s, got %s", model.ID(), wrapped.ID())
		}
	})

	t.Run("middleware can transform input", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Hello", testutil.MockUsage(10, 5)),
		})

		temp := 0.5
		middleware := &DefaultSettingsMiddleware{
			DefaultTemperature: &temp,
		}

		wrapped := WrapModel(model, middleware)

		_, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Check that temperature was applied
		if len(model.DoStreamCalls) != 1 {
			t.Fatalf("expected 1 call, got %d", len(model.DoStreamCalls))
		}

		call := model.DoStreamCalls[0]
		if call.Temperature == nil || *call.Temperature != 0.5 {
			t.Errorf("expected temperature 0.5, got %v", call.Temperature)
		}
	})

	t.Run("multiple middlewares are applied in order", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Hello", testutil.MockUsage(10, 5)),
		})

		var order []string

		middleware1 := &testMiddleware{
			name: "first",
			onTransform: func() {
				order = append(order, "first-transform")
			},
			onWrap: func() {
				order = append(order, "first-wrap")
			},
		}

		middleware2 := &testMiddleware{
			name: "second",
			onTransform: func() {
				order = append(order, "second-transform")
			},
			onWrap: func() {
				order = append(order, "second-wrap")
			},
		}

		wrapped := WrapModel(model, middleware1, middleware2)

		_, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// First middleware is outermost, so it transforms first and wraps last
		// Order: first-transform -> first-wrap (outer) -> second-transform -> second-wrap (inner) -> model
		expected := []string{"first-transform", "first-wrap", "second-transform", "second-wrap"}
		if len(order) != len(expected) {
			t.Fatalf("expected %d operations, got %d: %v", len(expected), len(order), order)
		}

		for i, exp := range expected {
			if order[i] != exp {
				t.Errorf("position %d: expected %s, got %s", i, exp, order[i])
			}
		}
	})
}

func TestDefaultSettingsMiddleware(t *testing.T) {
	t.Run("applies default temperature", func(t *testing.T) {
		temp := 0.7
		middleware := &DefaultSettingsMiddleware{
			DefaultTemperature: &temp,
		}

		options := &stream.CallOptions{}
		result := middleware.TransformOptions(options)

		if result.Temperature == nil || *result.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %v", result.Temperature)
		}
	})

	t.Run("does not override existing temperature", func(t *testing.T) {
		defaultTemp := 0.7
		existingTemp := 0.9
		middleware := &DefaultSettingsMiddleware{
			DefaultTemperature: &defaultTemp,
		}

		options := &stream.CallOptions{Temperature: &existingTemp}
		result := middleware.TransformOptions(options)

		if *result.Temperature != 0.9 {
			t.Errorf("expected temperature 0.9, got %v", *result.Temperature)
		}
	})

	t.Run("applies default provider options", func(t *testing.T) {
		middleware := &DefaultSettingsMiddleware{
			DefaultProviderOptions: map[string]any{
				"option1": "value1",
				"option2": "value2",
			},
		}

		options := &stream.CallOptions{
			ProviderOptions: map[string]any{
				"option1": "existing",
			},
		}

		result := middleware.TransformOptions(options)

		// option1 should not be overridden
		if result.ProviderOptions["option1"] != "existing" {
			t.Errorf("option1 should not be overridden")
		}

		// option2 should be added
		if result.ProviderOptions["option2"] != "value2" {
			t.Errorf("option2 should be added")
		}
	})
}

func TestLoggingMiddleware(t *testing.T) {
	t.Run("calls OnCall and OnEvent", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Hello", testutil.MockUsage(10, 5)),
		})

		var called bool
		var eventCount int

		middleware := &LoggingMiddleware{
			ModelID: "test-model",
			OnCall: func(modelID string, options *stream.CallOptions) {
				called = true
				if modelID != "test-model" {
					t.Errorf("expected model ID 'test-model', got '%s'", modelID)
				}
			},
			OnEvent: func(modelID string, event stream.Event) {
				eventCount++
			},
		}

		wrapped := WrapModel(model, middleware)

		events, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if !called {
			t.Error("OnCall was not called")
		}

		if eventCount == 0 {
			t.Error("OnEvent was not called")
		}
	})

	t.Run("calls OnError on error", func(t *testing.T) {
		model := &errorModel{err: errors.New("test error")}

		var errorCalled bool
		middleware := &LoggingMiddleware{
			ModelID: "test-model",
			OnError: func(modelID string, err error) {
				errorCalled = true
			},
		}

		wrapped := WrapModel(model, middleware)

		_, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err == nil {
			t.Fatal("expected error")
		}

		if !errorCalled {
			t.Error("OnError was not called")
		}
	})
}

func TestRetryMiddleware(t *testing.T) {
	t.Run("retries on transient errors", func(t *testing.T) {
		attempts := 0
		model := &retryTestModel{
			attemptsBeforeSuccess: 2,
			onAttempt: func() {
				attempts++
			},
		}

		middleware := &RetryMiddleware{
			MaxRetries: 3,
		}

		wrapped := WrapModel(model, middleware)

		_, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if attempts != 3 {
			t.Errorf("expected 3 attempts, got %d", attempts)
		}
	})

	t.Run("stops retrying when ShouldRetry returns false", func(t *testing.T) {
		attempts := 0
		model := &errorModel{err: errors.New("permanent error")}

		middleware := &RetryMiddleware{
			MaxRetries: 3,
			ShouldRetry: func(err error) bool {
				attempts++
				return false
			},
		}

		wrapped := WrapModel(model, middleware)

		_, err := wrapped.Stream(context.Background(), &stream.CallOptions{})
		if err == nil {
			t.Fatal("expected error")
		}

		if attempts != 1 {
			t.Errorf("expected 1 attempt, got %d", attempts)
		}
	})
}

// Test helpers

type testMiddleware struct {
	BaseMiddleware
	name        string
	onTransform func()
	onWrap      func()
}

func (t *testMiddleware) TransformOptions(options *stream.CallOptions) *stream.CallOptions {
	if t.onTransform != nil {
		t.onTransform()
	}
	return options
}

func (t *testMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	if t.onWrap != nil {
		t.onWrap()
	}
	return doStream(ctx, options)
}

type errorModel struct {
	err error
}

func (e *errorModel) ID() string       { return "error-model" }
func (e *errorModel) Provider() string { return "test" }
func (e *errorModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	return nil, e.err
}

type retryTestModel struct {
	attemptsBeforeSuccess int
	currentAttempt        int
	onAttempt             func()
}

func (r *retryTestModel) ID() string       { return "retry-model" }
func (r *retryTestModel) Provider() string { return "test" }
func (r *retryTestModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	if r.onAttempt != nil {
		r.onAttempt()
	}
	r.currentAttempt++
	if r.currentAttempt <= r.attemptsBeforeSuccess {
		return nil, errors.New("transient error")
	}

	// Success - return a simple event stream
	events := make(chan stream.Event, 1)
	events <- stream.Event{
		Type: stream.EventFinish,
		Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop},
	}
	close(events)
	return events, nil
}
