package middleware

import (
	"context"
	"strconv"

	"github.com/airlockrun/goai/stream"
)

// ExtractReasoningMiddleware intercepts streamed text, splits
// <TagName>...</TagName> sections out as typed reasoning events, and emits
// the remaining text normally. Useful for models that embed reasoning
// inline as plain text with XML-style tags (DeepSeek-R1 via chat
// completions, some OSS reasoners, etc.) instead of native reasoning
// events.
//
// Mirrors ai-sdk's extractReasoningMiddleware.
//
// Implementation note: the buffer-scan pattern from ai-sdk is preserved —
// tags may arrive split across stream deltas, so we use
// getPotentialStartIndex to detect partial matches and wait for more bytes
// before deciding.
type ExtractReasoningMiddleware struct {
	BaseMiddleware

	// TagName is the XML tag enclosing reasoning content (e.g. "think").
	// Required.
	TagName string

	// Separator is inserted between consecutive reasoning or consecutive
	// text sections that span a tag boundary. Defaults to "\n".
	Separator string

	// StartWithReasoning treats the stream as beginning inside the
	// reasoning tag. Useful for models that omit the opening tag because
	// reasoning is guaranteed to come first.
	StartWithReasoning bool
}

// WrapStream processes each TextStart..TextEnd pair as a single
// reasoning-extraction block.
func (m *ExtractReasoningMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	inner, err := doStream(ctx, options)
	if err != nil {
		return nil, err
	}

	openingTag := "<" + m.TagName + ">"
	closingTag := "</" + m.TagName + ">"
	separator := m.Separator
	if separator == "" {
		separator = "\n"
	}

	out := make(chan stream.Event, 16)
	go func() {
		defer close(out)

		type blockState struct {
			isFirstReasoning bool
			isFirstText      bool
			afterSwitch      bool
			isReasoning      bool
			buffer           string
			idCounter        int
			textStartEmitted bool
			reasoningOpen    bool
		}

		var st *blockState
		var pendingTextStart stream.Event

		// publish emits `text` as either a text-delta or a reasoning-delta
		// depending on the current mode. It handles the separator prefix
		// and the first-* bookkeeping identically to ai-sdk.
		publish := func(text string) {
			if text == "" || st == nil {
				return
			}
			prefix := ""
			if st.afterSwitch {
				if st.isReasoning && !st.isFirstReasoning {
					prefix = separator
				} else if !st.isReasoning && !st.isFirstText {
					prefix = separator
				}
			}

			if st.isReasoning {
				if st.afterSwitch || st.isFirstReasoning {
					id := "reasoning-" + strconv.Itoa(st.idCounter)
					out <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: id}}
					st.reasoningOpen = true
				}
				id := "reasoning-" + strconv.Itoa(st.idCounter)
				out <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: id, Text: prefix + text}}
			} else {
				if !st.textStartEmitted {
					out <- pendingTextStart
					st.textStartEmitted = true
				}
				out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: prefix + text}}
			}

			st.afterSwitch = false
			if st.isReasoning {
				st.isFirstReasoning = false
			} else {
				st.isFirstText = false
			}
		}

		for ev := range inner {
			switch ev.Type {
			case stream.EventTextStart:
				st = &blockState{
					isFirstReasoning: true,
					isFirstText:      true,
					isReasoning:      m.StartWithReasoning,
				}
				pendingTextStart = ev
				// Don't emit yet — a leading reasoning block delays text-start.
			case stream.EventTextDelta:
				if st == nil {
					out <- ev
					continue
				}
				st.buffer += ev.Data.(stream.TextDeltaEvent).Text
				for {
					nextTag := openingTag
					if st.isReasoning {
						nextTag = closingTag
					}
					startIdx := getPotentialStartIndex(st.buffer, nextTag)
					if startIdx < 0 {
						publish(st.buffer)
						st.buffer = ""
						break
					}
					// Publish any content before the (potential) tag.
					publish(st.buffer[:startIdx])

					foundFull := startIdx+len(nextTag) <= len(st.buffer)
					if foundFull {
						st.buffer = st.buffer[startIdx+len(nextTag):]
						// Flip state. If we were in reasoning, close it first.
						if st.isReasoning {
							id := "reasoning-" + strconv.Itoa(st.idCounter)
							// Empty reasoning block: no delta was published, so
							// reasoning-start was never emitted. Emit it now so
							// the consumer sees a balanced start/end pair.
							// Mirrors ai-sdk PR #12055.
							if st.isFirstReasoning {
								out <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: id}}
							}
							out <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: id}}
							st.reasoningOpen = false
							st.idCounter++
						}
						st.isReasoning = !st.isReasoning
						st.afterSwitch = true
					} else {
						// Partial match — keep waiting for more bytes.
						st.buffer = st.buffer[startIdx:]
						break
					}
				}
			case stream.EventTextEnd:
				if st == nil {
					out <- ev
					continue
				}
				// Flush any remaining non-tag content.
				if st.buffer != "" {
					publish(st.buffer)
					st.buffer = ""
				}
				if st.isReasoning && !st.reasoningOpen {
					// Ended inside an empty reasoning block — emit balanced
					// start+end so downstream isn't left with a dangling end.
					id := "reasoning-" + strconv.Itoa(st.idCounter)
					out <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: id}}
					out <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: id}}
				} else if st.reasoningOpen {
					id := "reasoning-" + strconv.Itoa(st.idCounter)
					out <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: id}}
					st.reasoningOpen = false
				}
				if st.textStartEmitted {
					out <- ev
				}
				st = nil
			default:
				out <- ev
			}
		}
	}()
	return out, nil
}
