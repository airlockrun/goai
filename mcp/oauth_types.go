package mcp

import (
	"fmt"
	"net/url"
)

// OAuth schema types mirroring ai-sdk's oauth-types.ts. Field names match
// the wire shape (snake_case JSON) so json.Unmarshal does the heavy lifting.

// OAuthTokens is the OAuth 2.1 token response.
type OAuthTokens struct {
	AccessToken  string `json:"access_token"`
	IDToken      string `json:"id_token,omitempty"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in,omitempty"`
	Scope        string `json:"scope,omitempty"`
	RefreshToken string `json:"refresh_token,omitempty"`
}

// AuthorizationServerMetadata is the union of OAuth 2.0 metadata (RFC 8414)
// and OpenID Connect Discovery 1.0 metadata. We carry the superset in a
// single struct; consumers check non-zero fields before use.
type AuthorizationServerMetadata struct {
	Issuer                                string   `json:"issuer"`
	AuthorizationEndpoint                 string   `json:"authorization_endpoint"`
	TokenEndpoint                         string   `json:"token_endpoint"`
	RegistrationEndpoint                  string   `json:"registration_endpoint,omitempty"`
	ScopesSupported                       []string `json:"scopes_supported,omitempty"`
	ResponseTypesSupported                []string `json:"response_types_supported"`
	GrantTypesSupported                   []string `json:"grant_types_supported,omitempty"`
	CodeChallengeMethodsSupported         []string `json:"code_challenge_methods_supported,omitempty"`
	TokenEndpointAuthMethodsSupported     []string `json:"token_endpoint_auth_methods_supported,omitempty"`
	TokenEndpointAuthSigningAlgValuesSupp []string `json:"token_endpoint_auth_signing_alg_values_supported,omitempty"`

	// OIDC-only fields (populated when discovery returned an OIDC document).
	UserinfoEndpoint                 string   `json:"userinfo_endpoint,omitempty"`
	JwksURI                          string   `json:"jwks_uri,omitempty"`
	SubjectTypesSupported            []string `json:"subject_types_supported,omitempty"`
	IDTokenSigningAlgValuesSupported []string `json:"id_token_signing_alg_values_supported,omitempty"`
	ClaimsSupported                  []string `json:"claims_supported,omitempty"`
}

// OAuthProtectedResourceMetadata is RFC 9728 protected-resource metadata.
type OAuthProtectedResourceMetadata struct {
	Resource                              string   `json:"resource"`
	AuthorizationServers                  []string `json:"authorization_servers,omitempty"`
	JwksURI                               string   `json:"jwks_uri,omitempty"`
	ScopesSupported                       []string `json:"scopes_supported,omitempty"`
	BearerMethodsSupported                []string `json:"bearer_methods_supported,omitempty"`
	ResourceSigningAlgValuesSupported     []string `json:"resource_signing_alg_values_supported,omitempty"`
	ResourceName                          string   `json:"resource_name,omitempty"`
	ResourceDocumentation                 string   `json:"resource_documentation,omitempty"`
	ResourcePolicyURI                     string   `json:"resource_policy_uri,omitempty"`
	ResourceTosURI                        string   `json:"resource_tos_uri,omitempty"`
	TLSClientCertificateBoundAccessTokens bool     `json:"tls_client_certificate_bound_access_tokens,omitempty"`
	AuthorizationDetailsTypesSupported    []string `json:"authorization_details_types_supported,omitempty"`
	DPopSigningAlgValuesSupported         []string `json:"dpop_signing_alg_values_supported,omitempty"`
	DPopBoundAccessTokensRequired         bool     `json:"dpop_bound_access_tokens_required,omitempty"`
}

// OAuthClientInformation is the registered-client identity (id + secret).
type OAuthClientInformation struct {
	ClientID              string `json:"client_id"`
	ClientSecret          string `json:"client_secret,omitempty"`
	ClientIDIssuedAt      int64  `json:"client_id_issued_at,omitempty"`
	ClientSecretExpiresAt int64  `json:"client_secret_expires_at,omitempty"`
}

// OAuthClientMetadata is the registration request body shape (RFC 7591).
type OAuthClientMetadata struct {
	RedirectURIs            []string `json:"redirect_uris"`
	TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method,omitempty"`
	GrantTypes              []string `json:"grant_types,omitempty"`
	ResponseTypes           []string `json:"response_types,omitempty"`
	ClientName              string   `json:"client_name,omitempty"`
	ClientURI               string   `json:"client_uri,omitempty"`
	LogoURI                 string   `json:"logo_uri,omitempty"`
	Scope                   string   `json:"scope,omitempty"`
	Contacts                []string `json:"contacts,omitempty"`
	TosURI                  string   `json:"tos_uri,omitempty"`
	PolicyURI               string   `json:"policy_uri,omitempty"`
	JwksURI                 string   `json:"jwks_uri,omitempty"`
	Jwks                    any      `json:"jwks,omitempty"`
	SoftwareID              string   `json:"software_id,omitempty"`
	SoftwareVersion         string   `json:"software_version,omitempty"`
	SoftwareStatement       string   `json:"software_statement,omitempty"`
}

// OAuthClientInformationFull combines registration response fields. It
// embeds both OAuthClientMetadata (the request body the server echoes back)
// and OAuthClientInformation (id/secret it issues).
type OAuthClientInformationFull struct {
	OAuthClientMetadata
	OAuthClientInformation
}

// OAuthErrorResponse is the body shape of a token/registration error per
// RFC 6749 §5.2.
type OAuthErrorResponse struct {
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description,omitempty"`
	ErrorURI         string `json:"error_uri,omitempty"`
}

// validateSafeURL rejects URL schemes that are unsafe in a redirect-style
// context, mirroring ai-sdk's SafeUrlSchema. javascript:, data:, and
// vbscript: would let a malicious metadata document smuggle code into a
// browser-resolved redirect.
func validateSafeURL(s string) error {
	u, err := url.Parse(s)
	if err != nil {
		return fmt.Errorf("parse url %q: %w", s, err)
	}
	switch u.Scheme {
	case "javascript", "data", "vbscript":
		return fmt.Errorf("url scheme %q not allowed", u.Scheme)
	}
	return nil
}
