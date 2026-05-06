package mcp

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
)

// generatePKCE creates an RFC 7636 PKCE pair (S256 method). The verifier is
// 32 random bytes encoded as URL-safe base64 (43 chars, within the spec's
// 43–128 range). The challenge is the URL-safe base64 of SHA-256(verifier).
func generatePKCE() (verifier, challenge string, err error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", "", fmt.Errorf("read random bytes: %w", err)
	}
	verifier = base64.RawURLEncoding.EncodeToString(buf)

	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return verifier, challenge, nil
}
