package mcp

import "fmt"

// OAuth errors mirroring ai-sdk's error/oauth-error.ts. Callers that need to
// branch on error kind use errors.As against the typed wrappers.

// MCPClientOAuthError is the base type for all OAuth-flow errors raised by
// this package. Code is the OAuth 2.0 error code from RFC 6749 §5.2 (or
// "server_error" / a synthetic code when we couldn't parse the response).
type MCPClientOAuthError struct {
	Code    string
	Message string
	Cause   error
}

func (e *MCPClientOAuthError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("oauth %s: %s", e.Code, e.Message)
	}
	return fmt.Sprintf("oauth %s", e.Code)
}

func (e *MCPClientOAuthError) Unwrap() error { return e.Cause }

// ServerError corresponds to OAuth "server_error".
type ServerError struct{ MCPClientOAuthError }

// InvalidClientError corresponds to OAuth "invalid_client".
type InvalidClientError struct{ MCPClientOAuthError }

// InvalidGrantError corresponds to OAuth "invalid_grant".
type InvalidGrantError struct{ MCPClientOAuthError }

// UnauthorizedClientError corresponds to OAuth "unauthorized_client".
type UnauthorizedClientError struct{ MCPClientOAuthError }

// UnauthorizedError signals that auth failed and the caller should treat
// the request as unauthenticated. Returned by transports when a 401 cannot
// be recovered from (no provider, refresh failed, second 401, etc.).
type UnauthorizedError struct{ Message string }

func (e *UnauthorizedError) Error() string {
	if e.Message == "" {
		return "unauthorized"
	}
	return e.Message
}

// newOAuthError constructs the right typed error for an OAuth error code.
// Unknown codes fall through as ServerError so callers always get an
// MCPClientOAuthError-shaped value.
func newOAuthError(code, description string, cause error) error {
	base := MCPClientOAuthError{Code: code, Message: description, Cause: cause}
	switch code {
	case "server_error":
		return &ServerError{base}
	case "invalid_client":
		return &InvalidClientError{base}
	case "invalid_grant":
		return &InvalidGrantError{base}
	case "unauthorized_client":
		return &UnauthorizedClientError{base}
	default:
		return &ServerError{base}
	}
}
