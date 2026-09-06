package internal

import (
	"bufio"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
)

type failingReader struct {
	data string
	err  error
	sent bool
}

func (r *failingReader) Read(p []byte) (int, error) {
	if !r.sent {
		r.sent = true
		return copy(p, r.data), nil
	}
	return 0, r.err
}

func TestStreamReaderErr(t *testing.T) {
	readErr := errors.New("connection reset")

	tests := []struct {
		name      string
		ctx       func() context.Context
		reader    io.Reader
		configure func(*bufio.Scanner)
		want      error
		apiError  bool
	}{
		{
			name:   "clean EOF",
			ctx:    context.Background,
			reader: strings.NewReader("complete\n"),
		},
		{
			name:     "transport read failure",
			ctx:      context.Background,
			reader:   &failingReader{data: "partial\n", err: readErr},
			want:     readErr,
			apiError: true,
		},
		{
			name: "cancellation",
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			reader: &failingReader{err: readErr},
			want:   context.Canceled,
		},
		{
			name:   "reader cancellation",
			ctx:    context.Background,
			reader: &failingReader{err: context.Canceled},
			want:   context.Canceled,
		},
		{
			name:   "scanner token too long",
			ctx:    context.Background,
			reader: strings.NewReader(strings.Repeat("x", bufio.MaxScanTokenSize+1)),
			configure: func(scanner *bufio.Scanner) {
				scanner.Buffer(make([]byte, 16), bufio.MaxScanTokenSize)
			},
			apiError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			streamReader := NewStreamReader(tt.reader)
			scanner := bufio.NewScanner(streamReader)
			if tt.configure != nil {
				tt.configure(scanner)
			}
			for scanner.Scan() {
			}

			err := streamReader.Err(tt.ctx(), scanner.Err())
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("Err() = %v, want %v", err, tt.want)
			}
			if tt.want == nil && tt.name == "clean EOF" && err != nil {
				t.Fatalf("Err() = %v, want nil", err)
			}

			var apiErr *goaierrors.APICallError
			if got := errors.As(err, &apiErr); got != tt.apiError {
				t.Fatalf("APICallError = %v, want %v (error: %v)", got, tt.apiError, err)
			}
			if tt.apiError && (!apiErr.IsRetryable || apiErr.Cause != readErr) {
				t.Errorf("APICallError = %+v, want retryable with original cause", apiErr)
			}
			if tt.name == "scanner token too long" && err == nil {
				t.Fatal("Err() = nil, want scanner framing error")
			}
		})
	}
}
