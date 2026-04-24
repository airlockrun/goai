package middleware

import (
	"context"

	"github.com/airlockrun/goai/stream"
)

// SimulateStreamingMiddleware consumes the inner stream in full, accumulates
// the contents, and re-emits them as a clean, coalesced stream — one
// TextDelta per text block, one ReasoningDelta per reasoning block, tool
// calls passed through in order.
//
// Mirrors ai-sdk's simulateStreamingMiddleware. Its canonical use is
// wrapping providers that return all content at once; for providers that
// already stream natively, it's a no-op that still flattens per-token
// chunking into per-block chunks. Test fixtures use this to get
// deterministic event ordering.
type SimulateStreamingMiddleware struct {
	BaseMiddleware
}

// WrapStream consumes the inner channel, buffers content, then replays it
// as a synthetic stream. Errors on the inner stream propagate verbatim.
func (s *SimulateStreamingMiddleware) WrapStream(ctx context.Context, options *stream.CallOptions, doStream StreamFunc) (<-chan stream.Event, error) {
	inner, err := doStream(ctx, options)
	if err != nil {
		return nil, err
	}

	// Drain the inner stream while preserving arrival order of distinct
	// items (text blocks, reasoning blocks, tool calls). Finish state is
	// tracked once at the end.
	type item struct {
		kind             string // "text" | "reasoning" | "toolcall" | "toolerror" | "toolresult" | "other"
		text             string
		reasoningID      string
		toolCall         stream.ToolCallEvent
		toolError        stream.ToolErrorEvent
		toolResult       stream.ToolResultEvent
		otherEvent       stream.Event
		providerMetadata map[string]any
	}

	var (
		items            []*item
		curText          *item
		curReasoning     map[string]*item
		finishReason     stream.FinishReason
		usage            stream.Usage
		providerMetadata map[string]any
		errEvent         *stream.ErrorEvent
	)
	curReasoning = make(map[string]*item)

	for ev := range inner {
		switch ev.Type {
		case stream.EventStart, stream.EventStartStep:
			// Drop — we re-emit these ourselves.
		case stream.EventTextStart:
			curText = &item{kind: "text"}
			items = append(items, curText)
		case stream.EventTextDelta:
			if curText == nil {
				curText = &item{kind: "text"}
				items = append(items, curText)
			}
			curText.text += ev.Data.(stream.TextDeltaEvent).Text
		case stream.EventTextEnd:
			curText = nil
		case stream.EventReasoningStart:
			r := ev.Data.(stream.ReasoningStartEvent)
			it := &item{kind: "reasoning", reasoningID: r.ID, providerMetadata: r.ProviderMetadata}
			curReasoning[r.ID] = it
			items = append(items, it)
		case stream.EventReasoningDelta:
			r := ev.Data.(stream.ReasoningDeltaEvent)
			if it, ok := curReasoning[r.ID]; ok {
				it.text += r.Text
			} else {
				// Delta without a prior start — synthesize.
				it := &item{kind: "reasoning", reasoningID: r.ID, text: r.Text}
				curReasoning[r.ID] = it
				items = append(items, it)
			}
		case stream.EventReasoningEnd:
			r := ev.Data.(stream.ReasoningEndEvent)
			delete(curReasoning, r.ID)
		case stream.EventToolCall:
			items = append(items, &item{kind: "toolcall", toolCall: ev.Data.(stream.ToolCallEvent)})
		case stream.EventToolResult:
			items = append(items, &item{kind: "toolresult", toolResult: ev.Data.(stream.ToolResultEvent)})
		case stream.EventToolError:
			items = append(items, &item{kind: "toolerror", toolError: ev.Data.(stream.ToolErrorEvent)})
		case stream.EventFinishStep, stream.EventFinish:
			// Capture on the last occurrence (Finish takes precedence).
			switch d := ev.Data.(type) {
			case stream.FinishStepEvent:
				finishReason = d.FinishReason
				usage = d.Usage
				if d.ProviderMetadata != nil {
					providerMetadata = d.ProviderMetadata
				}
			case stream.FinishEvent:
				finishReason = d.FinishReason
				usage = d.Usage
				if d.ProviderMetadata != nil {
					providerMetadata = d.ProviderMetadata
				}
			}
		case stream.EventError:
			if e, ok := ev.Data.(stream.ErrorEvent); ok {
				ee := e
				errEvent = &ee
			}
		default:
			// Preserve unknown/auxiliary events in order so callers using
			// non-core event types still get them through.
			items = append(items, &item{kind: "other", otherEvent: ev})
		}
	}

	out := make(chan stream.Event, 16)
	go func() {
		defer close(out)

		out <- stream.Event{Type: stream.EventStart, Data: stream.StartEvent{}}
		out <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}

		if errEvent != nil {
			out <- stream.Event{Type: stream.EventError, Data: *errEvent}
			return
		}

		for _, it := range items {
			switch it.kind {
			case "text":
				if it.text == "" {
					continue
				}
				out <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				out <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: it.text}}
				out <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
			case "reasoning":
				out <- stream.Event{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{ID: it.reasoningID, ProviderMetadata: it.providerMetadata}}
				if it.text != "" {
					out <- stream.Event{Type: stream.EventReasoningDelta, Data: stream.ReasoningDeltaEvent{ID: it.reasoningID, Text: it.text}}
				}
				out <- stream.Event{Type: stream.EventReasoningEnd, Data: stream.ReasoningEndEvent{ID: it.reasoningID}}
			case "toolcall":
				out <- stream.Event{Type: stream.EventToolCall, Data: it.toolCall}
			case "toolresult":
				out <- stream.Event{Type: stream.EventToolResult, Data: it.toolResult}
			case "toolerror":
				out <- stream.Event{Type: stream.EventToolError, Data: it.toolError}
			case "other":
				out <- it.otherEvent
			}
		}

		out <- stream.Event{Type: stream.EventFinishStep, Data: stream.FinishStepEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		}}
		out <- stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{
			FinishReason:     finishReason,
			Usage:            usage,
			ProviderMetadata: providerMetadata,
		}}
	}()
	return out, nil
}
