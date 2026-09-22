package mcp

// Protocol identifiers for OAuth metadata discovery and HTTP diagnostics.
// Session negotiation is owned by the official MCP SDK.

const (
	LatestProtocolVersion = "2026-07-28"

	HeaderProtocolVersion = "mcp-protocol-version"
	HeaderSessionID       = "mcp-session-id"
	HeaderLastEventID     = "last-event-id"

	UserAgentSuffix = "goai-mcp/1.0"
)
