package xai

import "github.com/airlockrun/goai/stream"

// mapXaiResponsesFinishReason maps xAI's finish/status string to goai's
// unified FinishReason. When the raw reason is ambiguous (e.g. "stop",
// "completed") and a function-call has been seen, the reason is
// overridden to tool-calls. Mirrors ai-sdk's map-xai-responses-finish-reason.ts.
func mapXaiResponsesFinishReason(reason string, hasFunctionCall bool) stream.FinishReason {
	switch reason {
	case "stop", "completed":
		if hasFunctionCall {
			return stream.FinishReasonToolCalls
		}
		return stream.FinishReasonStop
	case "length", "max_output_tokens":
		return stream.FinishReasonLength
	case "tool_calls", "function_call":
		return stream.FinishReasonToolCalls
	case "content_filter":
		return stream.FinishReasonContentFilter
	case "":
		// No reason provided — fall back to tool-calls when a function call
		// was seen, otherwise "other".
		if hasFunctionCall {
			return stream.FinishReasonToolCalls
		}
		return stream.FinishReasonOther
	default:
		return stream.FinishReasonOther
	}
}
