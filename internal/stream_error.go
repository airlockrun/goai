package internal

import (
	"context"
	"errors"
	"io"

	goaierrors "github.com/airlockrun/goai/errors"
)

// StreamReader records errors returned by the underlying streaming transport.
type StreamReader struct {
	reader io.Reader
	err    error
}

// NewStreamReader wraps a streaming response body for scanner error classification.
func NewStreamReader(reader io.Reader) *StreamReader {
	return &StreamReader{reader: reader}
}

func (r *StreamReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if err != nil && !errors.Is(err, io.EOF) && r.err == nil {
		r.err = err
	}
	return n, err
}

// Err classifies a scanner failure without treating scanner framing errors as transport failures.
func (r *StreamReader) Err(ctx context.Context, scannerErr error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if scannerErr == nil || r.err == nil || !errors.Is(scannerErr, r.err) {
		return scannerErr
	}
	if errors.Is(r.err, context.Canceled) || errors.Is(r.err, context.DeadlineExceeded) {
		return r.err
	}
	return goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{
		Message:        "stream read failed: " + r.err.Error(),
		Cause:          r.err,
		IsRetryable:    true,
		IsRetryableSet: true,
	})
}
