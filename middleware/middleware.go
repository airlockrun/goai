// Package middleware provides composable model transformations.
// This mirrors the ai-sdk middleware module.
package middleware

import (
	"context"

	"github.com/airlockrun/goai/stream"
)

// Middleware defines the interface for language model middleware.
// Middleware can transform parameters and wrap stream operations.
type Middleware interface {
	// TransformOptions transforms the options before they're passed to the model.
	// Return nil to pass through unchanged.
	TransformOptions(options *stream.CallOptions) *stream.CallOptions

	// WrapStream wraps the stream operation.
	// If nil, the stream passes through unchanged.
	WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error)
}

// StreamFunc is the function signature for stream operations.
type StreamFunc func(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error)

// WrapModel wraps a model with middleware.
// Multiple middlewares are applied in order: first middleware transforms options first,
// last middleware wraps the stream directly around the model.
func WrapModel(model stream.Model, middlewares ...Middleware) stream.Model {
	if len(middlewares) == 0 {
		return model
	}

	wrapped := model
	// Apply in reverse order so first middleware is outermost
	for i := len(middlewares) - 1; i >= 0; i-- {
		wrapped = &wrappedModel{
			inner:      wrapped,
			middleware: middlewares[i],
		}
	}
	return wrapped
}

// wrappedModel implements stream.Model with middleware applied.
type wrappedModel struct {
	inner      stream.Model
	middleware Middleware
}

func (w *wrappedModel) ID() string {
	return w.inner.ID()
}

func (w *wrappedModel) Provider() string {
	return w.inner.Provider()
}

func (w *wrappedModel) Stream(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
	// Transform options
	transformedOptions := w.middleware.TransformOptions(options)
	if transformedOptions == nil {
		transformedOptions = options
	}

	// Wrap stream
	doStream := func(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
		return w.inner.Stream(ctx, options)
	}

	return w.middleware.WrapStream(ctx, transformedOptions, doStream)
}

// BaseMiddleware provides a simple implementation that can be embedded.
// Override the methods you need.
type BaseMiddleware struct{}

// TransformOptions passes through unchanged by default.
func (b *BaseMiddleware) TransformOptions(options *stream.CallOptions) *stream.CallOptions {
	return options
}

// WrapStream passes through unchanged by default.
func (b *BaseMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	return doStream(ctx, options)
}

// DefaultSettingsMiddleware applies default settings to options.
type DefaultSettingsMiddleware struct {
	BaseMiddleware

	// DefaultTemperature is applied if options.Temperature is nil.
	DefaultTemperature *float64

	// DefaultTopP is applied if options.TopP is nil.
	DefaultTopP *float64

	// DefaultMaxOutputTokens is applied if options.MaxOutputTokens is nil.
	DefaultMaxOutputTokens *int

	// DefaultProviderOptions are merged with options.ProviderOptions.
	DefaultProviderOptions map[string]any
}

// TransformOptions applies default settings.
func (d *DefaultSettingsMiddleware) TransformOptions(options *stream.CallOptions) *stream.CallOptions {
	// Make a copy to avoid mutating the original
	result := *options

	if result.Temperature == nil && d.DefaultTemperature != nil {
		result.Temperature = d.DefaultTemperature
	}

	if result.TopP == nil && d.DefaultTopP != nil {
		result.TopP = d.DefaultTopP
	}

	if result.MaxOutputTokens == nil && d.DefaultMaxOutputTokens != nil {
		result.MaxOutputTokens = d.DefaultMaxOutputTokens
	}

	if len(d.DefaultProviderOptions) > 0 {
		if result.ProviderOptions == nil {
			result.ProviderOptions = make(map[string]any)
		}
		for k, v := range d.DefaultProviderOptions {
			if _, exists := result.ProviderOptions[k]; !exists {
				result.ProviderOptions[k] = v
			}
		}
	}

	return &result
}

// LoggingMiddleware logs model calls for debugging.
type LoggingMiddleware struct {
	BaseMiddleware

	// OnCall is called when the model is invoked.
	OnCall func(modelID string, options *stream.CallOptions)

	// OnEvent is called for each stream event.
	OnEvent func(modelID string, event stream.Event)

	// OnError is called when an error occurs.
	OnError func(modelID string, err error)

	// ModelID stores the model ID for logging.
	ModelID string
}

// WrapStream adds logging around the stream.
func (l *LoggingMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	if l.OnCall != nil {
		l.OnCall(l.ModelID, options)
	}

	events, err := doStream(ctx, options)
	if err != nil {
		if l.OnError != nil {
			l.OnError(l.ModelID, err)
		}
		return nil, err
	}

	// Wrap the events channel to log each event
	loggedEvents := make(chan stream.Event, 100)
	go func() {
		defer close(loggedEvents)
		for event := range events {
			if l.OnEvent != nil {
				l.OnEvent(l.ModelID, event)
			}
			loggedEvents <- event
		}
	}()

	return loggedEvents, nil
}

// RetryMiddleware adds automatic retry on transient errors.
type RetryMiddleware struct {
	BaseMiddleware

	// MaxRetries is the maximum number of retry attempts (default 3).
	MaxRetries int

	// ShouldRetry determines if an error should be retried.
	// Default: retry on any error.
	ShouldRetry func(error) bool
}

// WrapStream adds retry logic around the stream.
func (r *RetryMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	maxRetries := r.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}

	shouldRetry := r.ShouldRetry
	if shouldRetry == nil {
		shouldRetry = func(error) bool { return true }
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		events, err := doStream(ctx, options)
		if err == nil {
			return events, nil
		}

		lastErr = err
		if !shouldRetry(err) {
			return nil, err
		}

		// Check context before retrying
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}

	return nil, lastErr
}
