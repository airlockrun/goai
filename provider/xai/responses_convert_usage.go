package xai

import "github.com/airlockrun/goai/stream"

// convertXaiResponsesUsage converts xAI's Responses-API usage payload to
// goai's stream.Usage v3 shape. Mirrors ai-sdk's convertXaiResponsesUsage
// (references/ai-sdk/packages/xai/src/responses/convert-xai-responses-usage.ts).
//
// xAI reports usage via:
//
//	input_tokens
//	output_tokens
//	total_tokens
//	input_tokens_details.cached_tokens
//	output_tokens_details.reasoning_tokens
//
// The cached_tokens count is already-subsumed by input_tokens when
// cached_tokens <= input_tokens; otherwise the API reports them
// separately and we add them to the total to produce a sensible grand.
// reasoning_tokens is subtracted from output_tokens to recover the
// text-only output slice.
func convertXaiResponsesUsage(usage *responsesUsage) stream.Usage {
	if usage == nil {
		return stream.Usage{}
	}

	var cacheRead int
	var hasCacheRead bool
	if usage.InputTokensDetails != nil {
		cacheRead = usage.InputTokensDetails.CachedTokens
		hasCacheRead = true
	}

	var reasoning int
	var hasReasoning bool
	if usage.OutputTokensDetails != nil {
		reasoning = usage.OutputTokensDetails.ReasoningTokens
		hasReasoning = true
	}

	var inputTotal int
	var noCache int
	if cacheRead <= usage.InputTokens {
		inputTotal = usage.InputTokens
		noCache = usage.InputTokens - cacheRead
	} else {
		inputTotal = usage.InputTokens + cacheRead
		noCache = usage.InputTokens
	}

	out := stream.Usage{
		InputTokens: stream.InputTokens{
			Total:   stream.IntPtr(inputTotal),
			NoCache: stream.IntPtr(noCache),
		},
		OutputTokens: stream.OutputTokens{
			Total: stream.IntPtr(usage.OutputTokens),
			Text:  stream.IntPtr(usage.OutputTokens - reasoning),
		},
	}
	if hasCacheRead {
		out.InputTokens.CacheRead = stream.IntPtr(cacheRead)
	}
	if hasReasoning {
		out.OutputTokens.Reasoning = stream.IntPtr(reasoning)
	}
	return out
}
