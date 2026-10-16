package testutil

// AnthropicCachedInputUsageJSON reports uncached, cache-read and aggregate
// cache-write counts, including the write TTL breakdown.
const AnthropicCachedInputUsageJSON = `{"input_tokens":100,"cache_read_input_tokens":80,"cache_creation_input_tokens":20,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":12,"ephemeral_1h_input_tokens":8}}`

// AnthropicCachedMessageSSE is an offline Messages response requesting noop.
// Its inclusive usage is 200 input tokens and 42 output tokens.
const AnthropicCachedMessageSSE = `data: {"type":"message_start","message":{"usage":` + AnthropicCachedInputUsageJSON + `}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c","name":"noop","input":{}}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42,"output_tokens_by_type":{"text":40,"reasoning":2}}}

data: {"type":"message_stop"}

`
