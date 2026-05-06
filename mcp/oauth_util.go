package mcp

import (
	"fmt"
	"net/url"
	"strings"
)

// resourceURLFromServerURL strips the fragment per RFC 8707 §2 ("resource
// URIs MUST NOT include a fragment component"). All other components stay
// untouched.
func resourceURLFromServerURL(s string) (*url.URL, error) {
	u, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("parse server url: %w", err)
	}
	u.Fragment = ""
	return u, nil
}

// resourceURLStripSlash serializes a resource URL, removing the trailing
// slash that would otherwise appear on pathless URLs. The MCP spec says
// implementations SHOULD use the form without the trailing slash for
// interop.
func resourceURLStripSlash(u *url.URL) string {
	s := u.String()
	if (u.Path == "" || u.Path == "/") && strings.HasSuffix(s, "/") {
		return strings.TrimSuffix(s, "/")
	}
	return s
}

// checkResourceAllowed reports whether requested is a same-origin subpath
// of configured. Origin must match exactly; the requested path must equal
// or be a subpath of configured. We normalize both paths with a trailing
// "/" before prefix-matching to avoid "/api123" matching "/api".
func checkResourceAllowed(requested, configured string) (bool, error) {
	r, err := url.Parse(requested)
	if err != nil {
		return false, fmt.Errorf("parse requested: %w", err)
	}
	c, err := url.Parse(configured)
	if err != nil {
		return false, fmt.Errorf("parse configured: %w", err)
	}

	if originOf(r) != originOf(c) {
		return false, nil
	}

	if len(r.Path) < len(c.Path) {
		return false, nil
	}

	rp := r.Path
	if !strings.HasSuffix(rp, "/") {
		rp += "/"
	}
	cp := c.Path
	if !strings.HasSuffix(cp, "/") {
		cp += "/"
	}
	return strings.HasPrefix(rp, cp), nil
}

// originOf returns the origin (scheme://host[:port]) for u.
func originOf(u *url.URL) string {
	return u.Scheme + "://" + u.Host
}
