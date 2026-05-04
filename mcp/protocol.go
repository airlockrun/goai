package mcp

// Constants mirroring ai-sdk's mcp package for protocol negotiation +
// HTTP-transport headers. Keep in sync with
// references/ai-sdk/packages/mcp/src/tool/types.ts and mcp-http-transport.ts.

const (
	LatestProtocolVersion = "2025-11-25"

	HeaderProtocolVersion = "mcp-protocol-version"
	HeaderSessionID       = "mcp-session-id"
	HeaderLastEventID     = "last-event-id"

	UserAgentSuffix = "goai-mcp/1.0"
)

var SupportedProtocolVersions = []string{
	LatestProtocolVersion,
	"2025-06-18",
	"2025-03-26",
	"2024-11-05",
}
