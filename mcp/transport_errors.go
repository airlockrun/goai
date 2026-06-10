package mcp

// MCPClientError carries structured HTTP context for failures that
// originate from the streamable HTTP transport, so callers can branch on
// the response status without parsing the message string — e.g. to decide
// whether to fall back from HTTP to SSE transport on a 404/405 per the MCP
// spec. Mirrors ai-sdk's MCPClientError HTTP fields
// (references/ai-sdk/packages/mcp/src/error/mcp-client-error.ts).
//
// The fields are zero/empty for non-HTTP failures (network errors, parse
// errors without a response).
type MCPClientError struct {
	Message string

	// StatusCode is the HTTP status of the failing response. Zero when the
	// error has no associated response.
	StatusCode int

	// URL is the MCP endpoint the failing request was sent to.
	URL string

	// ResponseBody is the failing response body decoded as text, when read.
	ResponseBody string
}

func (e *MCPClientError) Error() string { return e.Message }
