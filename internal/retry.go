package internal

import (
	"context"
	"math/rand"
	"time"
)

// RetryConfig contains configuration for retry behavior.
type RetryConfig struct {
	// MaxAttempts is the maximum number of attempts (including the first).
	MaxAttempts int

	// InitialDelay is the initial delay before the first retry.
	InitialDelay time.Duration

	// MaxDelay is the maximum delay between retries.
	MaxDelay time.Duration

	// Multiplier is the factor by which the delay increases after each retry.
	Multiplier float64

	// Jitter adds randomness to the delay (0 to 1).
	Jitter float64

	// ShouldRetry determines if an error should trigger a retry.
	ShouldRetry func(err error, attempt int) bool
}

// DefaultRetryConfig returns a sensible default retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   2.0,
		Jitter:       0.2,
		ShouldRetry:  DefaultShouldRetry,
	}
}

// DefaultShouldRetry is the default retry predicate.
// It retries on transient errors (rate limits, server errors).
func DefaultShouldRetry(err error, attempt int) bool {
	if err == nil {
		return false
	}
	// Could check for specific error types here
	return true
}

// Retrier executes operations with retry logic.
type Retrier struct {
	config RetryConfig
}

// NewRetrier creates a new retrier with the given configuration.
func NewRetrier(config RetryConfig) *Retrier {
	if config.MaxAttempts <= 0 {
		config.MaxAttempts = 1
	}
	if config.InitialDelay <= 0 {
		config.InitialDelay = 1 * time.Second
	}
	if config.MaxDelay <= 0 {
		config.MaxDelay = 60 * time.Second
	}
	if config.Multiplier <= 0 {
		config.Multiplier = 2.0
	}
	if config.ShouldRetry == nil {
		config.ShouldRetry = DefaultShouldRetry
	}
	return &Retrier{config: config}
}

// Do executes the operation with retry logic.
func (r *Retrier) Do(ctx context.Context, op func() error) error {
	var lastErr error

	for attempt := 1; attempt <= r.config.MaxAttempts; attempt++ {
		err := op()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if we should retry
		if attempt >= r.config.MaxAttempts || !r.config.ShouldRetry(err, attempt) {
			return lastErr
		}

		// Calculate delay
		delay := r.calculateDelay(attempt)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}

	return lastErr
}

// DoWithResult executes an operation that returns a result with retry logic.
func (r *Retrier) DoWithResult(ctx context.Context, op func() (any, error)) (any, error) {
	var result any
	err := r.Do(ctx, func() error {
		var err error
		result, err = op()
		return err
	})
	return result, err
}

func (r *Retrier) calculateDelay(attempt int) time.Duration {
	delay := r.config.InitialDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * r.config.Multiplier)
		if delay > r.config.MaxDelay {
			delay = r.config.MaxDelay
			break
		}
	}

	// Add jitter
	if r.config.Jitter > 0 {
		jitter := float64(delay) * r.config.Jitter * (rand.Float64()*2 - 1)
		delay = time.Duration(float64(delay) + jitter)
		if delay < 0 {
			delay = 0
		}
	}

	return delay
}

// WithRetry is a convenience function for simple retry operations.
func WithRetry(ctx context.Context, maxAttempts int, op func() error) error {
	config := DefaultRetryConfig()
	config.MaxAttempts = maxAttempts
	return NewRetrier(config).Do(ctx, op)
}

// WithRetryResult is a convenience function for operations that return results.
func WithRetryResult[T any](ctx context.Context, maxAttempts int, op func() (T, error)) (T, error) {
	config := DefaultRetryConfig()
	config.MaxAttempts = maxAttempts

	var result T
	err := NewRetrier(config).Do(ctx, func() error {
		var err error
		result, err = op()
		return err
	})
	return result, err
}

// IsRetryable checks if an HTTP status code is retryable.
func IsRetryableStatus(statusCode int) bool {
	switch statusCode {
	case 429: // Too Many Requests
		return true
	case 500, 502, 503, 504: // Server errors
		return true
	default:
		return false
	}
}

// ParseRetryAfter parses the Retry-After header value.
// Returns the duration to wait, or 0 if not parseable.
func ParseRetryAfter(value string) time.Duration {
	if value == "" {
		return 0
	}

	// Try to parse as seconds
	var seconds int
	for _, c := range value {
		if c >= '0' && c <= '9' {
			seconds = seconds*10 + int(c-'0')
		} else {
			break
		}
	}

	if seconds > 0 {
		return time.Duration(seconds) * time.Second
	}

	// Could also try to parse as HTTP date, but that's complex
	return 0
}
