package internal

import (
	"bufio"
	"io"
	"strings"
)

// SSEEvent represents a Server-Sent Event.
type SSEEvent struct {
	// Event is the event type (from "event:" field).
	Event string

	// Data is the event data (from "data:" field).
	Data string

	// ID is the event ID (from "id:" field).
	ID string

	// Retry is the retry interval in milliseconds (from "retry:" field).
	Retry int
}

// SSEReader reads Server-Sent Events from a stream.
type SSEReader struct {
	scanner *bufio.Scanner
	err     error
}

// NewSSEReader creates a new SSE reader.
func NewSSEReader(r io.Reader) *SSEReader {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer
	return &SSEReader{scanner: scanner}
}

// Next reads the next SSE event.
// Returns nil when the stream ends or an error occurs.
func (r *SSEReader) Next() *SSEEvent {
	var event SSEEvent
	var dataLines []string
	hasData := false

	for r.scanner.Scan() {
		line := r.scanner.Text()

		// Empty line signals end of event
		if line == "" {
			if hasData {
				event.Data = strings.Join(dataLines, "\n")
				return &event
			}
			continue
		}

		// Parse field
		if strings.HasPrefix(line, "data:") {
			hasData = true
			data := strings.TrimPrefix(line, "data:")
			if len(data) > 0 && data[0] == ' ' {
				data = data[1:]
			}
			dataLines = append(dataLines, data)
		} else if strings.HasPrefix(line, "event:") {
			event.Event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "id:") {
			event.ID = strings.TrimSpace(strings.TrimPrefix(line, "id:"))
		} else if strings.HasPrefix(line, "retry:") {
			// Parse retry as integer
			retryStr := strings.TrimSpace(strings.TrimPrefix(line, "retry:"))
			var retry int
			for _, c := range retryStr {
				if c >= '0' && c <= '9' {
					retry = retry*10 + int(c-'0')
				} else {
					break
				}
			}
			event.Retry = retry
		}
		// Ignore comments (lines starting with :) and unknown fields
	}

	r.err = r.scanner.Err()
	return nil
}

// Err returns any error that occurred during reading.
func (r *SSEReader) Err() error {
	return r.err
}

// Events returns a channel that yields SSE events.
// The channel is closed when the stream ends or an error occurs.
func (r *SSEReader) Events() <-chan *SSEEvent {
	ch := make(chan *SSEEvent, 100)
	go func() {
		defer close(ch)
		for {
			event := r.Next()
			if event == nil {
				return
			}
			ch <- event
		}
	}()
	return ch
}

// SSEParser provides a simpler line-by-line SSE parsing interface.
type SSEParser struct {
	dataPrefix  string
	eventPrefix string
}

// NewSSEParser creates a new SSE parser with default prefixes.
func NewSSEParser() *SSEParser {
	return &SSEParser{
		dataPrefix:  "data: ",
		eventPrefix: "event: ",
	}
}

// ParseLine parses a single SSE line.
// Returns the data content if the line is a data line, empty string otherwise.
// Returns true for the second value if this is a data line.
func (p *SSEParser) ParseLine(line string) (string, bool) {
	if strings.HasPrefix(line, p.dataPrefix) {
		return strings.TrimPrefix(line, p.dataPrefix), true
	}
	if strings.HasPrefix(line, "data:") {
		return strings.TrimPrefix(line, "data:"), true
	}
	return "", false
}

// IsEventLine returns true if the line is an event type line.
func (p *SSEParser) IsEventLine(line string) bool {
	return strings.HasPrefix(line, p.eventPrefix) || strings.HasPrefix(line, "event:")
}

// GetEventType extracts the event type from an event line.
func (p *SSEParser) GetEventType(line string) string {
	if strings.HasPrefix(line, p.eventPrefix) {
		return strings.TrimPrefix(line, p.eventPrefix)
	}
	if strings.HasPrefix(line, "event:") {
		return strings.TrimSpace(strings.TrimPrefix(line, "event:"))
	}
	return ""
}

// IsDone returns true if the data is the SSE done marker.
func (p *SSEParser) IsDone(data string) bool {
	return data == "[DONE]"
}
