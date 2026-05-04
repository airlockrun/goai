package mcp

import (
	"bufio"
	"io"
	"strings"
)

// scanSSE parses a Server-Sent Events stream from r and invokes onEvent for
// each complete event (terminated by a blank line). eventType defaults to
// "message" when the stream omits an `event:` field, matching the
// EventSource specification. Multi-line `data:` fields are joined with "\n";
// the leading single space after `data:` (and `event:` / `id:`) is trimmed
// per spec.
//
// Mirrors the line-handling done by ai-sdk's EventSourceParserStream
// (consumed by both mcp-http-transport.ts and mcp-sse-transport.ts) so that
// callers see the same parsed shape as ai-sdk.
func scanSSE(r io.Reader, onEvent func(eventType, data, id string)) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var eventType string
	var dataLines []string
	var id string

	flush := func() {
		if len(dataLines) == 0 {
			eventType = ""
			id = ""
			return
		}
		t := eventType
		if t == "" {
			t = "message"
		}
		onEvent(t, strings.Join(dataLines, "\n"), id)
		eventType = ""
		dataLines = nil
		id = ""
	}

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, ":") {
			// Comment line per SSE spec.
			continue
		}

		field, value := splitSSEField(line)
		switch field {
		case "event":
			eventType = value
		case "data":
			dataLines = append(dataLines, value)
		case "id":
			id = value
		}
	}
	// EventSource only dispatches on blank lines, so any pending event when
	// the stream ends without a trailing blank is discarded. Mirror that.
	return scanner.Err()
}

// splitSSEField splits "field: value" into (field, value), trimming the
// single optional space after the colon as the SSE spec dictates. A line
// without a colon is treated as a field name with empty value.
func splitSSEField(line string) (string, string) {
	idx := strings.IndexByte(line, ':')
	if idx < 0 {
		return line, ""
	}
	field := line[:idx]
	value := line[idx+1:]
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return field, value
}
