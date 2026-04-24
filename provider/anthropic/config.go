package anthropic

// Config configures how an AnthropicModel builds and dispatches requests.
// The zero value corresponds to the direct Anthropic API. Non-direct
// consumers (Bedrock Anthropic, Vertex Anthropic) supply hooks to
// override URL construction, transform the outgoing body, and advertise
// capability differences. Mirrors ai-sdk's AnthropicMessagesConfig
// (packages/anthropic/src/anthropic-messages-language-model.ts).
type Config struct {
	// ProviderID overrides the provider identifier reported by the model
	// (Model.Provider()). Empty string falls back to "anthropic".
	ProviderID string

	// BuildRequestURL returns the full URL for a request. baseURL is the
	// provider's configured base; streaming is true for streaming
	// requests. Nil falls back to the standard Anthropic "/messages"
	// endpoint.
	BuildRequestURL func(baseURL string, streaming bool) string

	// TransformRequestBody rewrites the assembled JSON body (as a generic
	// map) before it is marshalled. betas carries the set of collected
	// anthropic-beta tokens; the transform may return a merged slice or
	// otherwise rewrite the body (e.g. add anthropic_version, strip
	// model). Nil leaves the body unchanged.
	TransformRequestBody func(body map[string]any, betas []string) map[string]any

	// ToolBetaMap maps a tool wire-type to an anthropic_beta header
	// string. The builder merges hits into the request's collected beta
	// list. Nil leaves collected betas untouched.
	ToolBetaMap map[string]string

	// ToolsStrict forces strict:true on every function-tool definition in
	// the request body (Bedrock behavior, per ai-sdk #91f8777). Ignored
	// when SupportsStrictTools resolves to false.
	ToolsStrict bool

	// SupportsNativeStructuredOutput reports whether the provider accepts
	// the native structured-output format via output_config.format. When
	// false, the builder falls back to synthetic JSON-tool injection even
	// if the caller requested structuredOutputMode=outputFormat. Nil
	// defaults to true (direct Anthropic behavior).
	SupportsNativeStructuredOutput *bool

	// SupportsStrictTools reports whether tool definitions may carry
	// strict:true. When false, strict is stripped regardless of
	// ToolsStrict. Vertex Anthropic sets this to false. Nil defaults to
	// true.
	SupportsStrictTools *bool

	// SupportedURLs returns a map of media-type globs to URL-regex lists
	// that the provider accepts directly. Nil accepts any https URL,
	// matching the direct-Anthropic default.
	SupportedURLs func() map[string][]string

	// EmitBetasInBody controls whether the builder writes the collected
	// anthropic_beta list into the request body (Bedrock expectation).
	// When false, betas are left for the caller to place on the
	// `anthropic-beta` HTTP header (direct Anthropic / Vertex behavior).
	EmitBetasInBody bool
}

func (c Config) providerID() string {
	if c.ProviderID != "" {
		return c.ProviderID
	}
	return "anthropic"
}

func (c Config) supportsNativeStructuredOutput() bool {
	if c.SupportsNativeStructuredOutput == nil {
		return true
	}
	return *c.SupportsNativeStructuredOutput
}

func (c Config) supportsStrictTools() bool {
	if c.SupportsStrictTools == nil {
		return true
	}
	return *c.SupportsStrictTools
}
