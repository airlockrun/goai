package middleware

import (
	"context"
	"regexp"
	"strings"

	"github.com/airlockrun/goai/stream"
)

// ExtractJsonMiddleware strips markdown JSON code fences from streamed text
// so downstream structured-output parsers see raw JSON. Useful for models
// that wrap their JSON responses in ```json … ``` even when asked for
// strict JSON.
//
// Mirrors ai-sdk's extractJsonMiddleware.
//
// Two modes:
//   - Default (Transform == nil): streams content incrementally, stripping
//     the leading ```json\n once the first newline arrives and trimming
//     the trailing ``` at the end via a tail-buffer.
//   - Custom Transform: buffers the entire text block and applies Transform
//     at text-end. Emits one text-delta with the transformed payload.
type ExtractJsonMiddleware struct {
	BaseMiddleware

	// Transform is an optional custom text transformer. When nil, the
	// default markdown-fence stripper is used.
	Transform func(string) string
}

var (
	jsonFencePrefix = regexp.MustCompile("^```(?:json)?\\s*\\n")
	jsonFenceSuffix = regexp.MustCompile("\\n?```\\s*$")
)

// defaultJSONStrip strips leading ```json\n and trailing ``` fences.
func defaultJSONStrip(s string) string {
	s = jsonFencePrefix.ReplaceAllString(s, "")
	s = jsonFenceSuffix.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// WrapStream intercepts text deltas between TextStart and TextEnd and
// strips fences. Non-text events pass through untouched.
func (m *ExtractJsonMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	inner, err := doStream(ctx, options)
	if err != nil {
		return nil, err
	}

	hasCustom := m.Transform != nil
	transform := m.Transform
	if transform == nil {
		transform = defaultJSONStrip
	}

	out := make(chan stream.Event, 16)
	go func() {
		defer close(out)

		type phase int
		const (
			phaseIdle phase = iota
			// phasePrefix is the opening window where we're determining
			// whether the stream starts with a markdown fence.
			phasePrefix
			// phaseStreaming emits transformed content incrementally.
			phaseStreaming
			// phaseBuffering holds all content until text-end (custom transform).
			phaseBuffering
		)
		const suffixBufferSize = 12

		var (
			cur            phase
			buffer         string
			prefixStripped bool
			startEmitted   bool
			startPending   stream.Event
		)

		flushStreaming := func(final bool) {
			if cur != phaseStreaming {
				return
			}
			if final {
				remaining := buffer
				if prefixStripped {
					remaining = jsonFenceSuffix.ReplaceAllString(remaining, "")
					remaining = strings.TrimRight(remaining, " \t\r\n")
				} else {
					remaining = transform(remaining)
				}
				buffer = ""
				if remaining != "" {
					out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: remaining}}
				}
				return
			}
			if len(buffer) > suffixBufferSize {
				emit := buffer[:len(buffer)-suffixBufferSize]
				buffer = buffer[len(buffer)-suffixBufferSize:]
				out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: emit}}
			}
		}

		for ev := range inner {
			switch ev.Type {
			case stream.EventTextStart:
				startPending = ev
				startEmitted = false
				buffer = ""
				prefixStripped = false
				if hasCustom {
					cur = phaseBuffering
				} else {
					cur = phasePrefix
				}
			case stream.EventTextDelta:
				if cur == phaseIdle {
					out <- ev
					continue
				}
				buffer += ev.Data.(stream.TextDeltaEvent).Text
				if cur == phaseBuffering {
					continue
				}
				if cur == phasePrefix {
					if len(buffer) > 0 && !strings.HasPrefix(buffer, "`") {
						cur = phaseStreaming
						if !startEmitted {
							out <- startPending
							startEmitted = true
						}
					} else if strings.HasPrefix(buffer, "```") {
						if strings.Contains(buffer, "\n") {
							if loc := jsonFencePrefix.FindStringIndex(buffer); loc != nil {
								buffer = buffer[loc[1]:]
								prefixStripped = true
							}
							// Either way we move past prefix detection.
							cur = phaseStreaming
							if !startEmitted {
								out <- startPending
								startEmitted = true
							}
						}
						// else keep buffering until we see a newline.
					} else if len(buffer) >= 3 && !strings.HasPrefix(buffer, "```") {
						cur = phaseStreaming
						if !startEmitted {
							out <- startPending
							startEmitted = true
						}
					}
				}
				flushStreaming(false)
			case stream.EventTextEnd:
				if !startEmitted && cur != phaseIdle {
					// Emit the deferred TextStart now so the Start/End pair
					// stays symmetrical even when buffer was empty or custom.
					out <- startPending
					startEmitted = true
				}
				switch cur {
				case phaseBuffering:
					if buffer != "" {
						text := transform(buffer)
						buffer = ""
						if text != "" {
							out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}}
						}
					}
				case phaseStreaming:
					flushStreaming(true)
				case phasePrefix:
					// Stream ended before we left prefix detection. Whatever
					// sits in buffer gets the full transform.
					if buffer != "" {
						text := transform(buffer)
						buffer = ""
						if text != "" {
							out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: text}}
						}
					}
				}
				out <- ev
				cur = phaseIdle
			default:
				out <- ev
			}
		}
	}()
	return out, nil
}
