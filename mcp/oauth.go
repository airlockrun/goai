package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

// OAuth flow port of references/ai-sdk/packages/mcp/src/tool/oauth.ts.
// Public surface mirrors ai-sdk's: callers implement OAuthClientProvider,
// pass it to the transport, and the transport runs Auth(...) on a 401.

// AuthResult is the return value of Auth — either the caller is ready to
// retry (AuthResultAuthorized) or they need to wait for a redirect-driven
// authorization code to come back (AuthResultRedirect).
type AuthResult string

const (
	AuthResultAuthorized AuthResult = "AUTHORIZED"
	AuthResultRedirect   AuthResult = "REDIRECT"
)

// OAuthClientProvider is the integration point for OAuth-protected MCP
// servers. Mirrors ai-sdk's OAuthClientProvider — callers implement it to
// own token storage, redirect handling, and (optionally) dynamic client
// registration.
type OAuthClientProvider interface {
	Tokens(ctx context.Context) (*OAuthTokens, error)
	SaveTokens(ctx context.Context, t *OAuthTokens) error
	RedirectToAuthorization(ctx context.Context, authURL *url.URL) error
	SaveCodeVerifier(ctx context.Context, v string) error
	CodeVerifier(ctx context.Context) (string, error)
	RedirectURL() string
	ClientMetadata() OAuthClientMetadata
	ClientInformation(ctx context.Context) (*OAuthClientInformation, error)
}

// ClientAuthenticator opts into custom client authentication on token
// requests, replacing the default Basic/POST/none selection logic.
type ClientAuthenticator interface {
	AddClientAuthentication(ctx context.Context, headers http.Header, params url.Values, tokenURL *url.URL, metadata *AuthorizationServerMetadata) error
}

// CredentialInvalidator opts into local cleanup when the server reports
// credentials are no longer valid (so the user doesn't have to intervene).
type CredentialInvalidator interface {
	// Scope is one of "all", "client", "tokens", "verifier".
	InvalidateCredentials(ctx context.Context, scope string) error
}

// ClientInformationSaver opts the provider into dynamic registration —
// without it, registerClient cannot persist the issued client_id.
type ClientInformationSaver interface {
	SaveClientInformation(ctx context.Context, info *OAuthClientInformationFull) error
}

// StateGenerator opts into CSRF state on the authorization redirect.
type StateGenerator interface {
	State(ctx context.Context) (string, error)
	SaveState(ctx context.Context, s string) error
	StoredState(ctx context.Context) (string, error)
}

// ResourceURLValidator opts into custom resource-URL validation, replacing
// the default same-origin check against the protected-resource metadata.
type ResourceURLValidator interface {
	ValidateResourceURL(ctx context.Context, serverURL, resource string) (*url.URL, error)
}

// resourceMetadataRegex extracts the resource_metadata="..." parameter
// from a WWW-Authenticate header (RFC 9728).
var resourceMetadataRegex = regexp.MustCompile(`resource_metadata="([^"]*)"`)

// ExtractResourceMetadataURL pulls the protected-resource-metadata URL
// from a 401 response's WWW-Authenticate header. Returns nil when the
// header is missing, not bearer, or has no resource_metadata parameter.
func ExtractResourceMetadataURL(resp *http.Response) *url.URL {
	header := resp.Header.Get("Www-Authenticate")
	if header == "" {
		header = resp.Header.Get("WWW-Authenticate")
	}
	if header == "" {
		return nil
	}
	parts := strings.SplitN(header, " ", 2)
	if len(parts) < 2 || !strings.EqualFold(parts[0], "bearer") {
		return nil
	}
	m := resourceMetadataRegex.FindStringSubmatch(header)
	if m == nil {
		return nil
	}
	u, err := url.Parse(m[1])
	if err != nil {
		return nil
	}
	return u
}

// httpDoer is the subset of http.Client used by the OAuth flow. Tests
// substitute a mock; production code passes nil for the package default.
type httpDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

func defaultHTTPClient(c httpDoer) httpDoer {
	if c != nil {
		return c
	}
	return http.DefaultClient
}

// buildWellKnownPath constructs the well-known path for auth metadata
// discovery. When prependPathname is true (OIDC-style), the well-known
// segment goes after the issuer path; otherwise it's at the root.
func buildWellKnownPath(prefix, pathname string, prependPathname bool) string {
	if strings.HasSuffix(pathname, "/") {
		pathname = strings.TrimSuffix(pathname, "/")
	}
	if prependPathname {
		return pathname + "/.well-known/" + prefix
	}
	return "/.well-known/" + prefix + pathname
}

// tryMetadataDiscovery does a single GET against u with the protocol-
// version header, returning nil response on network errors (mirroring
// ai-sdk's CORS-fallback semantics; in Go we only see the error case).
func tryMetadataDiscovery(ctx context.Context, client httpDoer, u *url.URL, protocolVersion string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("MCP-Protocol-Version", protocolVersion)
	resp, err := defaultHTTPClient(client).Do(req)
	if err != nil {
		// ai-sdk falls back to a no-headers retry on TypeError (browser CORS).
		// In Go we don't have that signal; surface the error.
		return nil, err
	}
	return resp, nil
}

func shouldAttemptFallback(resp *http.Response, pathname string) bool {
	if resp == nil {
		return true
	}
	return resp.StatusCode >= 400 && resp.StatusCode < 500 && pathname != "/"
}

// discoverMetadataWithFallback runs the well-known lookup with the issuer
// pathname then falls back to the root if needed.
func discoverMetadataWithFallback(
	ctx context.Context,
	client httpDoer,
	serverURL string,
	wellKnownType string,
	protocolVersion string,
	metadataURL string,
) (*http.Response, error) {
	issuer, err := url.Parse(serverURL)
	if err != nil {
		return nil, fmt.Errorf("parse server url: %w", err)
	}

	var target *url.URL
	if metadataURL != "" {
		target, err = url.Parse(metadataURL)
		if err != nil {
			return nil, fmt.Errorf("parse metadata url: %w", err)
		}
	} else {
		path := buildWellKnownPath(wellKnownType, issuer.Path, false)
		target = &url.URL{Scheme: issuer.Scheme, Host: issuer.Host, Path: path, RawQuery: issuer.RawQuery}
	}

	resp, err := tryMetadataDiscovery(ctx, client, target, protocolVersion)
	if err != nil && metadataURL != "" {
		return nil, err
	}

	if metadataURL == "" && shouldAttemptFallback(resp, issuer.Path) {
		if resp != nil {
			resp.Body.Close()
		}
		root := &url.URL{Scheme: issuer.Scheme, Host: issuer.Host, Path: "/.well-known/" + wellKnownType}
		resp, err = tryMetadataDiscovery(ctx, client, root, protocolVersion)
		if err != nil {
			return nil, err
		}
	}
	return resp, nil
}

// discoverOAuthProtectedResourceMetadata fetches RFC 9728 metadata.
func discoverOAuthProtectedResourceMetadata(
	ctx context.Context,
	client httpDoer,
	serverURL string,
	resourceMetadataURL string,
	protocolVersion string,
) (*OAuthProtectedResourceMetadata, error) {
	if protocolVersion == "" {
		protocolVersion = LatestProtocolVersion
	}
	resp, err := discoverMetadataWithFallback(ctx, client, serverURL, "oauth-protected-resource", protocolVersion, resourceMetadataURL)
	if err != nil {
		return nil, err
	}
	if resp == nil || resp.StatusCode == http.StatusNotFound {
		if resp != nil {
			resp.Body.Close()
		}
		return nil, errors.New("resource server does not implement OAuth 2.0 Protected Resource Metadata")
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d loading protected resource metadata", resp.StatusCode)
	}
	var meta OAuthProtectedResourceMetadata
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return nil, fmt.Errorf("decode protected resource metadata: %w", err)
	}
	return &meta, nil
}

// discoveryEndpoint represents one of the well-known URLs we'll try.
type discoveryEndpoint struct {
	URL  *url.URL
	Type string // "oauth" or "oidc"
}

// buildDiscoveryURLs returns prioritized discovery URLs for an
// authorization server URL, mirroring oauth.ts:272-325.
func buildDiscoveryURLs(authServerURL string) ([]discoveryEndpoint, error) {
	u, err := url.Parse(authServerURL)
	if err != nil {
		return nil, fmt.Errorf("parse auth server url: %w", err)
	}
	hasPath := u.Path != "" && u.Path != "/"
	out := make([]discoveryEndpoint, 0, 4)

	if !hasPath {
		out = append(out,
			discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/.well-known/oauth-authorization-server"}, Type: "oauth"},
			discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/.well-known/openid-configuration"}, Type: "oidc"},
		)
		return out, nil
	}

	path := strings.TrimSuffix(u.Path, "/")
	out = append(out,
		discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/.well-known/oauth-authorization-server" + path}, Type: "oauth"},
		discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/.well-known/oauth-authorization-server"}, Type: "oauth"},
		discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/.well-known/openid-configuration" + path}, Type: "oidc"},
		discoveryEndpoint{URL: &url.URL{Scheme: u.Scheme, Host: u.Host, Path: path + "/.well-known/openid-configuration"}, Type: "oidc"},
	)
	return out, nil
}

// discoverAuthorizationServerMetadata walks the discovery URL list and
// returns the first valid metadata document. Returns nil with no error
// when no endpoint succeeded — callers fall back to legacy behavior.
func discoverAuthorizationServerMetadata(
	ctx context.Context,
	client httpDoer,
	authServerURL string,
	protocolVersion string,
) (*AuthorizationServerMetadata, error) {
	if protocolVersion == "" {
		protocolVersion = LatestProtocolVersion
	}
	endpoints, err := buildDiscoveryURLs(authServerURL)
	if err != nil {
		return nil, err
	}

	for _, ep := range endpoints {
		resp, err := tryMetadataDiscovery(ctx, client, ep.URL, protocolVersion)
		if err != nil {
			// Treat network errors like CORS failures: try the next endpoint.
			continue
		}
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			resp.Body.Close()
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body := readAndClose(resp)
			return nil, fmt.Errorf("HTTP %d loading %s metadata from %s: %s", resp.StatusCode, ep.Type, ep.URL, body)
		}

		var meta AuthorizationServerMetadata
		if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("decode %s metadata: %w", ep.Type, err)
		}
		resp.Body.Close()

		if ep.Type == "oidc" {
			// MCP spec requires OIDC providers to support S256 PKCE.
			if !containsString(meta.CodeChallengeMethodsSupported, "S256") {
				return nil, fmt.Errorf("incompatible OIDC provider at %s: does not support S256 code challenge", ep.URL)
			}
		}
		return &meta, nil
	}
	return nil, nil
}

// startAuthorizationOptions groups the inputs to startAuthorization to
// avoid an unwieldy positional signature.
type startAuthorizationOptions struct {
	Metadata          *AuthorizationServerMetadata
	ClientInformation *OAuthClientInformation
	RedirectURL       string
	Scope             string
	State             string
	Resource          *url.URL
}

// startAuthorization builds the redirect URL for the user-agent flow and
// returns the URL plus the PKCE verifier (which the caller must persist
// to exchange later).
func startAuthorization(authServerURL string, opts startAuthorizationOptions) (*url.URL, string, error) {
	const responseType = "code"
	const codeChallengeMethod = "S256"

	var authURL *url.URL
	if opts.Metadata != nil {
		u, err := url.Parse(opts.Metadata.AuthorizationEndpoint)
		if err != nil {
			return nil, "", fmt.Errorf("parse authorization_endpoint: %w", err)
		}
		if !containsString(opts.Metadata.ResponseTypesSupported, responseType) {
			return nil, "", fmt.Errorf("incompatible auth server: does not support response type %s", responseType)
		}
		if !containsString(opts.Metadata.CodeChallengeMethodsSupported, codeChallengeMethod) {
			return nil, "", fmt.Errorf("incompatible auth server: does not support code challenge method %s", codeChallengeMethod)
		}
		authURL = u
	} else {
		base, err := url.Parse(authServerURL)
		if err != nil {
			return nil, "", fmt.Errorf("parse auth server url: %w", err)
		}
		authURL = base.ResolveReference(&url.URL{Path: "/authorize"})
	}

	verifier, challenge, err := generatePKCE()
	if err != nil {
		return nil, "", err
	}

	q := authURL.Query()
	q.Set("response_type", responseType)
	q.Set("client_id", opts.ClientInformation.ClientID)
	q.Set("code_challenge", challenge)
	q.Set("code_challenge_method", codeChallengeMethod)
	q.Set("redirect_uri", opts.RedirectURL)
	if opts.State != "" {
		q.Set("state", opts.State)
	}
	if opts.Scope != "" {
		q.Set("scope", opts.Scope)
	}
	if strings.Contains(opts.Scope, "offline_access") {
		// OIDC offline_access requires explicit consent prompt.
		q.Add("prompt", "consent")
	}
	if opts.Resource != nil {
		q.Set("resource", resourceURLStripSlash(opts.Resource))
	}
	authURL.RawQuery = q.Encode()
	return authURL, verifier, nil
}

// clientAuthMethod is one of "client_secret_basic" | "client_secret_post"
// | "none".
type clientAuthMethod string

const (
	authBasic clientAuthMethod = "client_secret_basic"
	authPost  clientAuthMethod = "client_secret_post"
	authNone  clientAuthMethod = "none"
)

// selectClientAuthMethod picks the best supported method given the
// server's advertised list and whether we have a client secret.
func selectClientAuthMethod(client *OAuthClientInformation, supported []string) clientAuthMethod {
	hasSecret := client.ClientSecret != ""
	if len(supported) == 0 {
		if hasSecret {
			return authPost
		}
		return authNone
	}
	if hasSecret && containsString(supported, "client_secret_basic") {
		return authBasic
	}
	if hasSecret && containsString(supported, "client_secret_post") {
		return authPost
	}
	if containsString(supported, "none") {
		return authNone
	}
	if hasSecret {
		return authPost
	}
	return authNone
}

// applyClientAuthentication mutates headers/params per the chosen method.
func applyClientAuthentication(method clientAuthMethod, client *OAuthClientInformation, headers http.Header, params url.Values) error {
	switch method {
	case authBasic:
		if client.ClientSecret == "" {
			return errors.New("client_secret_basic requires a client_secret")
		}
		creds := base64.StdEncoding.EncodeToString([]byte(client.ClientID + ":" + client.ClientSecret))
		headers.Set("Authorization", "Basic "+creds)
		return nil
	case authPost:
		params.Set("client_id", client.ClientID)
		if client.ClientSecret != "" {
			params.Set("client_secret", client.ClientSecret)
		}
		return nil
	case authNone:
		params.Set("client_id", client.ClientID)
		return nil
	default:
		return fmt.Errorf("unsupported client auth method: %s", method)
	}
}

// parseErrorResponse converts an OAuth error response (body + status) into
// a typed MCPClientOAuthError. Falls back to ServerError when the body
// isn't a valid OAuth error JSON.
func parseErrorResponse(body []byte, status int) error {
	var er OAuthErrorResponse
	if err := json.Unmarshal(body, &er); err == nil && er.Error != "" {
		return newOAuthError(er.Error, er.ErrorDescription, nil)
	}
	prefix := ""
	if status > 0 {
		prefix = fmt.Sprintf("HTTP %d: ", status)
	}
	return &ServerError{MCPClientOAuthError{
		Code:    "server_error",
		Message: prefix + "invalid OAuth error response. Raw body: " + string(body),
	}}
}

// exchangeAuthorizationOptions groups the inputs to exchangeAuthorization.
type exchangeAuthorizationOptions struct {
	Metadata          *AuthorizationServerMetadata
	ClientInformation *OAuthClientInformation
	AuthorizationCode string
	CodeVerifier      string
	RedirectURI       string
	Resource          *url.URL
	Authenticator     ClientAuthenticator
}

// exchangeAuthorization exchanges an authorization code for tokens.
func exchangeAuthorization(ctx context.Context, client httpDoer, authServerURL string, opts exchangeAuthorizationOptions) (*OAuthTokens, error) {
	const grantType = "authorization_code"
	tokenURL, err := resolveTokenURL(opts.Metadata, authServerURL)
	if err != nil {
		return nil, err
	}
	if opts.Metadata != nil && len(opts.Metadata.GrantTypesSupported) > 0 && !containsString(opts.Metadata.GrantTypesSupported, grantType) {
		return nil, fmt.Errorf("incompatible auth server: does not support grant type %s", grantType)
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/x-www-form-urlencoded")
	headers.Set("Accept", "application/json")

	params := url.Values{}
	params.Set("grant_type", grantType)
	params.Set("code", opts.AuthorizationCode)
	params.Set("code_verifier", opts.CodeVerifier)
	params.Set("redirect_uri", opts.RedirectURI)

	if err := applyAuthOrCustom(ctx, opts.Authenticator, opts.ClientInformation, opts.Metadata, headers, params, tokenURL); err != nil {
		return nil, err
	}
	if opts.Resource != nil {
		params.Set("resource", resourceURLStripSlash(opts.Resource))
	}

	return postTokenRequest(ctx, client, tokenURL.String(), headers, params, "")
}

// refreshAuthorizationOptions groups inputs to refreshAuthorization.
type refreshAuthorizationOptions struct {
	Metadata          *AuthorizationServerMetadata
	ClientInformation *OAuthClientInformation
	RefreshToken      string
	Resource          *url.URL
	Authenticator     ClientAuthenticator
}

// refreshAuthorization swaps a refresh token for a fresh access token.
func refreshAuthorization(ctx context.Context, client httpDoer, authServerURL string, opts refreshAuthorizationOptions) (*OAuthTokens, error) {
	const grantType = "refresh_token"
	tokenURL, err := resolveTokenURL(opts.Metadata, authServerURL)
	if err != nil {
		return nil, err
	}
	if opts.Metadata != nil && len(opts.Metadata.GrantTypesSupported) > 0 && !containsString(opts.Metadata.GrantTypesSupported, grantType) {
		return nil, fmt.Errorf("incompatible auth server: does not support grant type %s", grantType)
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/x-www-form-urlencoded")
	headers.Set("Accept", "application/json")

	params := url.Values{}
	params.Set("grant_type", grantType)
	params.Set("refresh_token", opts.RefreshToken)

	if err := applyAuthOrCustom(ctx, opts.Authenticator, opts.ClientInformation, opts.Metadata, headers, params, tokenURL); err != nil {
		return nil, err
	}
	if opts.Resource != nil {
		params.Set("resource", resourceURLStripSlash(opts.Resource))
	}

	// Preserve the original refresh_token if the server omits a new one
	// (RFC 6749 §6: refresh_token MAY be issued, otherwise reuse).
	return postTokenRequest(ctx, client, tokenURL.String(), headers, params, opts.RefreshToken)
}

// registerClient performs RFC 7591 dynamic client registration.
func registerClient(ctx context.Context, client httpDoer, authServerURL string, metadata *AuthorizationServerMetadata, clientMetadata OAuthClientMetadata) (*OAuthClientInformationFull, error) {
	var registrationURL string
	if metadata != nil {
		if metadata.RegistrationEndpoint == "" {
			return nil, errors.New("incompatible auth server: does not support dynamic client registration")
		}
		registrationURL = metadata.RegistrationEndpoint
	} else {
		base, err := url.Parse(authServerURL)
		if err != nil {
			return nil, fmt.Errorf("parse auth server url: %w", err)
		}
		registrationURL = base.ResolveReference(&url.URL{Path: "/register"}).String()
	}

	body, err := json.Marshal(clientMetadata)
	if err != nil {
		return nil, fmt.Errorf("marshal client metadata: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", registrationURL, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := defaultHTTPClient(client).Do(req)
	if err != nil {
		return nil, fmt.Errorf("registration request: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, parseErrorResponse(respBody, resp.StatusCode)
	}

	var info OAuthClientInformationFull
	if err := json.Unmarshal(respBody, &info); err != nil {
		return nil, fmt.Errorf("decode client information: %w", err)
	}
	return &info, nil
}

// AuthOptions controls a single Auth call.
type AuthOptions struct {
	ServerURL           string
	AuthorizationCode   string
	CallbackState       string
	Scope               string
	ResourceMetadataURL *url.URL
	HTTPClient          httpDoer
}

// Auth runs the OAuth flow once and returns whether the caller is ready
// to retry (AUTHORIZED) or needs to wait for the redirect (REDIRECT). Auth
// transparently retries authInternal once when refresh hits InvalidClient
// or InvalidGrant — mirrors oauth.ts:833-860.
func Auth(ctx context.Context, provider OAuthClientProvider, opts AuthOptions) (AuthResult, error) {
	res, err := authInternal(ctx, provider, opts)
	if err == nil {
		return res, nil
	}

	var invalidClient *InvalidClientError
	var unauthorizedClient *UnauthorizedClientError
	var invalidGrant *InvalidGrantError
	switch {
	case errors.As(err, &invalidClient) || errors.As(err, &unauthorizedClient):
		if inv, ok := provider.(CredentialInvalidator); ok {
			_ = inv.InvalidateCredentials(ctx, "all")
		}
		return authInternal(ctx, provider, opts)
	case errors.As(err, &invalidGrant):
		if inv, ok := provider.(CredentialInvalidator); ok {
			_ = inv.InvalidateCredentials(ctx, "tokens")
		}
		return authInternal(ctx, provider, opts)
	}
	return "", err
}

// selectResourceURL picks the resource URL to bind tokens to, validating
// against the protected-resource metadata when available.
func selectResourceURL(ctx context.Context, serverURL string, provider OAuthClientProvider, meta *OAuthProtectedResourceMetadata) (*url.URL, error) {
	defaultResource, err := resourceURLFromServerURL(serverURL)
	if err != nil {
		return nil, err
	}
	if v, ok := provider.(ResourceURLValidator); ok {
		var resource string
		if meta != nil {
			resource = meta.Resource
		}
		return v.ValidateResourceURL(ctx, defaultResource.String(), resource)
	}
	if meta == nil {
		return nil, nil
	}
	allowed, err := checkResourceAllowed(defaultResource.String(), meta.Resource)
	if err != nil {
		return nil, err
	}
	if !allowed {
		return nil, fmt.Errorf("protected resource %s does not match expected %s (or origin)", meta.Resource, defaultResource.String())
	}
	return url.Parse(meta.Resource)
}

// authInternal is the full flow body — see oauth.ts:893-1051. We deviate
// from ai-sdk only where Go's stricter typing makes the JS optional-method
// pattern awkward (see optional interfaces above).
func authInternal(ctx context.Context, provider OAuthClientProvider, opts AuthOptions) (AuthResult, error) {
	client := defaultHTTPClient(opts.HTTPClient)

	var resourceMeta *OAuthProtectedResourceMetadata
	var authServerURL string
	if rm, err := discoverOAuthProtectedResourceMetadata(ctx, client, opts.ServerURL, urlOrEmpty(opts.ResourceMetadataURL), LatestProtocolVersion); err == nil {
		resourceMeta = rm
		if len(rm.AuthorizationServers) > 0 {
			authServerURL = rm.AuthorizationServers[0]
		}
	}
	if authServerURL == "" {
		// Legacy MCP spec: server-as-auth-server fallback.
		authServerURL = opts.ServerURL
	}

	resource, err := selectResourceURL(ctx, opts.ServerURL, provider, resourceMeta)
	if err != nil {
		return "", err
	}

	metadata, err := discoverAuthorizationServerMetadata(ctx, client, authServerURL, LatestProtocolVersion)
	if err != nil {
		return "", err
	}

	clientInformation, err := provider.ClientInformation(ctx)
	if err != nil {
		return "", fmt.Errorf("client information: %w", err)
	}
	if clientInformation == nil {
		if opts.AuthorizationCode != "" {
			return "", errors.New("existing OAuth client information is required when exchanging an authorization code")
		}
		saver, ok := provider.(ClientInformationSaver)
		if !ok {
			return "", errors.New("OAuth client information must be saveable for dynamic registration")
		}
		full, err := registerClient(ctx, client, authServerURL, metadata, provider.ClientMetadata())
		if err != nil {
			return "", err
		}
		if err := saver.SaveClientInformation(ctx, full); err != nil {
			return "", fmt.Errorf("save client information: %w", err)
		}
		clientInformation = &full.OAuthClientInformation
	}

	auth, _ := provider.(ClientAuthenticator)

	if opts.AuthorizationCode != "" {
		if state, ok := provider.(StateGenerator); ok {
			expected, err := state.StoredState(ctx)
			if err == nil && expected != "" && expected != opts.CallbackState {
				return "", errors.New("OAuth state parameter mismatch - possible CSRF attack")
			}
		}
		verifier, err := provider.CodeVerifier(ctx)
		if err != nil {
			return "", fmt.Errorf("code verifier: %w", err)
		}
		tokens, err := exchangeAuthorization(ctx, client, authServerURL, exchangeAuthorizationOptions{
			Metadata:          metadata,
			ClientInformation: clientInformation,
			AuthorizationCode: opts.AuthorizationCode,
			CodeVerifier:      verifier,
			RedirectURI:       provider.RedirectURL(),
			Resource:          resource,
			Authenticator:     auth,
		})
		if err != nil {
			return "", err
		}
		if err := provider.SaveTokens(ctx, tokens); err != nil {
			return "", fmt.Errorf("save tokens: %w", err)
		}
		return AuthResultAuthorized, nil
	}

	tokens, err := provider.Tokens(ctx)
	if err != nil {
		return "", fmt.Errorf("tokens: %w", err)
	}
	if tokens != nil && tokens.RefreshToken != "" {
		newTokens, err := refreshAuthorization(ctx, client, authServerURL, refreshAuthorizationOptions{
			Metadata:          metadata,
			ClientInformation: clientInformation,
			RefreshToken:      tokens.RefreshToken,
			Resource:          resource,
			Authenticator:     auth,
		})
		if err == nil {
			if err := provider.SaveTokens(ctx, newTokens); err != nil {
				return "", fmt.Errorf("save tokens: %w", err)
			}
			return AuthResultAuthorized, nil
		}
		// Mirror oauth.ts:1016-1027: ServerError (or non-OAuth failure) →
		// swallow and fall through to a fresh authorization. Recoverable
		// OAuth errors (invalid_client, invalid_grant, unauthorized_client)
		// → re-raise so the outer Auth() wrapper can invalidate credentials
		// and retry once.
		var invClient *InvalidClientError
		var invGrant *InvalidGrantError
		var unauthClient *UnauthorizedClientError
		if errors.As(err, &invClient) || errors.As(err, &invGrant) || errors.As(err, &unauthClient) {
			return "", err
		}
		// ServerError or non-OAuth: continue to redirect flow.
	}

	state := ""
	if sg, ok := provider.(StateGenerator); ok {
		s, err := sg.State(ctx)
		if err != nil {
			return "", fmt.Errorf("state: %w", err)
		}
		state = s
		if state != "" {
			if err := sg.SaveState(ctx, state); err != nil {
				return "", fmt.Errorf("save state: %w", err)
			}
		}
	}

	scope := opts.Scope
	if scope == "" {
		scope = provider.ClientMetadata().Scope
	}
	authURL, verifier, err := startAuthorization(authServerURL, startAuthorizationOptions{
		Metadata:          metadata,
		ClientInformation: clientInformation,
		RedirectURL:       provider.RedirectURL(),
		Scope:             scope,
		State:             state,
		Resource:          resource,
	})
	if err != nil {
		return "", err
	}
	if err := provider.SaveCodeVerifier(ctx, verifier); err != nil {
		return "", fmt.Errorf("save code verifier: %w", err)
	}
	if err := provider.RedirectToAuthorization(ctx, authURL); err != nil {
		return "", fmt.Errorf("redirect to authorization: %w", err)
	}
	return AuthResultRedirect, nil
}

// resolveTokenURL returns the token endpoint, falling back to /token on
// the auth server URL when metadata is missing.
func resolveTokenURL(metadata *AuthorizationServerMetadata, authServerURL string) (*url.URL, error) {
	if metadata != nil && metadata.TokenEndpoint != "" {
		return url.Parse(metadata.TokenEndpoint)
	}
	base, err := url.Parse(authServerURL)
	if err != nil {
		return nil, fmt.Errorf("parse auth server url: %w", err)
	}
	return base.ResolveReference(&url.URL{Path: "/token"}), nil
}

// applyAuthOrCustom dispatches to the caller's ClientAuthenticator if one
// is provided, else picks a method from the metadata + client info.
func applyAuthOrCustom(ctx context.Context, custom ClientAuthenticator, client *OAuthClientInformation, metadata *AuthorizationServerMetadata, headers http.Header, params url.Values, tokenURL *url.URL) error {
	if custom != nil {
		return custom.AddClientAuthentication(ctx, headers, params, tokenURL, metadata)
	}
	var supported []string
	if metadata != nil {
		supported = metadata.TokenEndpointAuthMethodsSupported
	}
	method := selectClientAuthMethod(client, supported)
	return applyClientAuthentication(method, client, headers, params)
}

// postTokenRequest executes a token-endpoint POST and decodes the response.
// preserveRefresh, when non-empty, is the original refresh_token to fall
// back on when the server omits a new one (refresh flow).
func postTokenRequest(ctx context.Context, client httpDoer, tokenURL string, headers http.Header, params url.Values, preserveRefresh string) (*OAuthTokens, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", tokenURL, strings.NewReader(params.Encode()))
	if err != nil {
		return nil, err
	}
	for k, vs := range headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}

	resp, err := defaultHTTPClient(client).Do(req)
	if err != nil {
		return nil, fmt.Errorf("token request: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, parseErrorResponse(body, resp.StatusCode)
	}

	var tokens OAuthTokens
	if err := json.Unmarshal(body, &tokens); err != nil {
		return nil, fmt.Errorf("decode tokens: %w", err)
	}
	if tokens.RefreshToken == "" && preserveRefresh != "" {
		tokens.RefreshToken = preserveRefresh
	}
	return &tokens, nil
}

func containsString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func urlOrEmpty(u *url.URL) string {
	if u == nil {
		return ""
	}
	return u.String()
}

func readAndClose(resp *http.Response) string {
	if resp == nil {
		return ""
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return string(b)
}
