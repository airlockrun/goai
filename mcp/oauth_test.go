package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestExtractResourceMetadataURL(t *testing.T) {
	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"no header", "", ""},
		{"non-bearer", `Basic realm="x"`, ""},
		{"bearer no resource", `Bearer realm="x"`, ""},
		{"bearer with resource", `Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"`, "https://example.com/.well-known/oauth-protected-resource"},
		{"bearer with realm and resource", `Bearer realm="x", resource_metadata="https://a/b"`, "https://a/b"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{Header: http.Header{}}
			if tt.header != "" {
				resp.Header.Set("WWW-Authenticate", tt.header)
			}
			got := ExtractResourceMetadataURL(resp)
			if tt.want == "" {
				if got != nil {
					t.Errorf("got %v, want nil", got)
				}
				return
			}
			if got == nil || got.String() != tt.want {
				t.Errorf("got %v, want %s", got, tt.want)
			}
		})
	}
}

func TestSelectClientAuthMethod(t *testing.T) {
	tests := []struct {
		name      string
		hasSecret bool
		supported []string
		want      clientAuthMethod
	}{
		{"empty supported with secret", true, nil, authPost},
		{"empty supported no secret", false, nil, authNone},
		{"basic preferred when supported", true, []string{"client_secret_basic", "client_secret_post"}, authBasic},
		{"post fallback", true, []string{"client_secret_post"}, authPost},
		{"none for public client", false, []string{"none"}, authNone},
		{"secret + only none → none (server preference wins)", true, []string{"none"}, authNone},
		{"no secret + only basic → none", false, []string{"client_secret_basic"}, authNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OAuthClientInformation{ClientID: "id"}
			if tt.hasSecret {
				c.ClientSecret = "shh"
			}
			got := selectClientAuthMethod(c, tt.supported)
			if got != tt.want {
				t.Errorf("got %s, want %s", got, tt.want)
			}
		})
	}
}

func TestApplyClientAuthentication(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		h := http.Header{}
		p := url.Values{}
		err := applyClientAuthentication(authBasic, &OAuthClientInformation{ClientID: "abc", ClientSecret: "xyz"}, h, p)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		want := "Basic " + base64.StdEncoding.EncodeToString([]byte("abc:xyz"))
		if h.Get("Authorization") != want {
			t.Errorf("got %q, want %q", h.Get("Authorization"), want)
		}
		if p.Get("client_id") != "" {
			t.Error("basic auth must not set client_id in params")
		}
	})

	t.Run("basic without secret", func(t *testing.T) {
		err := applyClientAuthentication(authBasic, &OAuthClientInformation{ClientID: "abc"}, http.Header{}, url.Values{})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("post", func(t *testing.T) {
		h := http.Header{}
		p := url.Values{}
		_ = applyClientAuthentication(authPost, &OAuthClientInformation{ClientID: "abc", ClientSecret: "xyz"}, h, p)
		if p.Get("client_id") != "abc" || p.Get("client_secret") != "xyz" {
			t.Errorf("got params %v", p)
		}
	})

	t.Run("none", func(t *testing.T) {
		h := http.Header{}
		p := url.Values{}
		_ = applyClientAuthentication(authNone, &OAuthClientInformation{ClientID: "abc"}, h, p)
		if p.Get("client_id") != "abc" || p.Get("client_secret") != "" {
			t.Errorf("got params %v", p)
		}
	})
}

func TestParseErrorResponse(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantType string
		wantMsg  string
	}{
		{"invalid_grant", `{"error":"invalid_grant","error_description":"expired"}`, "*mcp.InvalidGrantError", "expired"},
		{"invalid_client", `{"error":"invalid_client"}`, "*mcp.InvalidClientError", ""},
		{"unknown code", `{"error":"weird_thing","error_description":"hm"}`, "*mcp.ServerError", "hm"},
		{"non-json", `not json`, "*mcp.ServerError", "Raw body: not json"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := parseErrorResponse([]byte(tt.body), 400)
			if err == nil {
				t.Fatal("expected error")
			}
			gotType := errType(err)
			if gotType != tt.wantType {
				t.Errorf("type %s, want %s", gotType, tt.wantType)
			}
			if tt.wantMsg != "" && !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("message %q missing %q", err.Error(), tt.wantMsg)
			}
		})
	}
}

func errType(err error) string {
	switch err.(type) {
	case *InvalidGrantError:
		return "*mcp.InvalidGrantError"
	case *InvalidClientError:
		return "*mcp.InvalidClientError"
	case *UnauthorizedClientError:
		return "*mcp.UnauthorizedClientError"
	case *ServerError:
		return "*mcp.ServerError"
	default:
		return "unknown"
	}
}

func TestPKCE(t *testing.T) {
	v, c, err := generatePKCE()
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 43 {
		t.Errorf("verifier length %d, want 43", len(v))
	}
	sum := sha256.Sum256([]byte(v))
	want := base64.RawURLEncoding.EncodeToString(sum[:])
	if c != want {
		t.Errorf("challenge mismatch:\n got %s\nwant %s", c, want)
	}
}

func TestResourceURLFromServerURL(t *testing.T) {
	u, err := resourceURLFromServerURL("https://example.com/api?x=1#frag")
	if err != nil {
		t.Fatal(err)
	}
	if u.String() != "https://example.com/api?x=1" {
		t.Errorf("got %s", u.String())
	}
}

func TestResourceURLStripSlash(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"https://example.com/", "https://example.com"},
		{"https://example.com/api/", "https://example.com/api/"},
		{"https://example.com/api", "https://example.com/api"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			u, _ := url.Parse(tt.in)
			got := resourceURLStripSlash(u)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCheckResourceAllowed(t *testing.T) {
	tests := []struct {
		req, conf string
		want      bool
	}{
		{"https://a.com/api/users", "https://a.com/api", true},
		{"https://a.com/api", "https://a.com/api", true},
		{"https://a.com/api123", "https://a.com/api", false},
		{"https://a.com/", "https://a.com/api", false},
		{"https://b.com/api/users", "https://a.com/api", false},
	}
	for _, tt := range tests {
		t.Run(tt.req, func(t *testing.T) {
			got, err := checkResourceAllowed(tt.req, tt.conf)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExchangeAuthorization(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			t.Errorf("path %s", r.URL.Path)
		}
		if r.Method != "POST" {
			t.Errorf("method %s", r.Method)
		}
		if err := r.ParseForm(); err != nil {
			t.Fatal(err)
		}
		if r.Form.Get("grant_type") != "authorization_code" {
			t.Errorf("grant_type %s", r.Form.Get("grant_type"))
		}
		if r.Form.Get("code") != "abc" || r.Form.Get("code_verifier") != "v" {
			t.Errorf("form %v", r.Form)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "tok", TokenType: "Bearer", RefreshToken: "rt"})
	}))
	defer srv.Close()

	tokens, err := exchangeAuthorization(context.Background(), nil, srv.URL, exchangeAuthorizationOptions{
		Metadata:          &AuthorizationServerMetadata{TokenEndpoint: srv.URL + "/oauth/token", ResponseTypesSupported: []string{"code"}},
		ClientInformation: &OAuthClientInformation{ClientID: "cid"},
		AuthorizationCode: "abc",
		CodeVerifier:      "v",
		RedirectURI:       "https://x/cb",
	})
	if err != nil {
		t.Fatal(err)
	}
	if tokens.AccessToken != "tok" || tokens.RefreshToken != "rt" {
		t.Errorf("tokens %+v", tokens)
	}
}

func TestRefreshAuthorization_PreservesOriginalRefreshToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Server omits refresh_token in response.
		json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "fresh", TokenType: "Bearer"})
	}))
	defer srv.Close()

	tokens, err := refreshAuthorization(context.Background(), nil, srv.URL, refreshAuthorizationOptions{
		Metadata:          &AuthorizationServerMetadata{TokenEndpoint: srv.URL + "/t"},
		ClientInformation: &OAuthClientInformation{ClientID: "cid"},
		RefreshToken:      "old-refresh",
	})
	if err != nil {
		t.Fatal(err)
	}
	if tokens.RefreshToken != "old-refresh" {
		t.Errorf("expected original refresh token preserved, got %q", tokens.RefreshToken)
	}
}

func TestRegisterClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(OAuthClientInformationFull{
			OAuthClientInformation: OAuthClientInformation{ClientID: "registered", ClientSecret: "s"},
		})
	}))
	defer srv.Close()
	info, err := registerClient(context.Background(), nil, srv.URL, &AuthorizationServerMetadata{RegistrationEndpoint: srv.URL + "/reg"}, OAuthClientMetadata{RedirectURIs: []string{"https://x/cb"}})
	if err != nil {
		t.Fatal(err)
	}
	if info.ClientID != "registered" || info.ClientSecret != "s" {
		t.Errorf("got %+v", info)
	}
}

func TestDiscoverAuthorizationServerMetadata_OAuth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(AuthorizationServerMetadata{
				Issuer:                        "http://" + r.Host,
				AuthorizationEndpoint:         "http://" + r.Host + "/auth",
				TokenEndpoint:                 "http://" + r.Host + "/token",
				ResponseTypesSupported:        []string{"code"},
				CodeChallengeMethodsSupported: []string{"S256"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	meta, err := discoverAuthorizationServerMetadata(context.Background(), nil, srv.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	if meta == nil || meta.TokenEndpoint == "" {
		t.Errorf("got %+v", meta)
	}
}

func TestDiscoverAuthorizationServerMetadata_OIDCRequiresS256(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Path == "/.well-known/openid-configuration" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(AuthorizationServerMetadata{
				Issuer:                           "issuer",
				AuthorizationEndpoint:            "https://x/auth",
				TokenEndpoint:                    "https://x/token",
				ResponseTypesSupported:           []string{"code"},
				CodeChallengeMethodsSupported:    nil, // missing S256
				JwksURI:                          "https://x/jwks",
				SubjectTypesSupported:            []string{"public"},
				IDTokenSigningAlgValuesSupported: []string{"RS256"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	_, err := discoverAuthorizationServerMetadata(context.Background(), nil, srv.URL, "")
	if err == nil || !strings.Contains(err.Error(), "S256") {
		t.Errorf("expected S256 error, got %v", err)
	}
}

// stubProvider is the minimal OAuthClientProvider for Auth tests.
type stubProvider struct {
	clientInfo  *OAuthClientInformation
	clientMeta  OAuthClientMetadata
	redirectURL string

	tokens         *OAuthTokens
	savedTokens    *OAuthTokens
	verifier       string
	savedVerifier  string
	state          string
	storedState    string
	savedState     string
	redirectedURL  *url.URL
	invalidations  []string
	savedClientInf *OAuthClientInformationFull
}

func (s *stubProvider) Tokens(ctx context.Context) (*OAuthTokens, error) { return s.tokens, nil }
func (s *stubProvider) SaveTokens(ctx context.Context, t *OAuthTokens) error {
	s.savedTokens = t
	return nil
}
func (s *stubProvider) RedirectToAuthorization(ctx context.Context, u *url.URL) error {
	s.redirectedURL = u
	return nil
}
func (s *stubProvider) SaveCodeVerifier(ctx context.Context, v string) error {
	s.savedVerifier = v
	return nil
}
func (s *stubProvider) CodeVerifier(ctx context.Context) (string, error) { return s.verifier, nil }
func (s *stubProvider) RedirectURL() string                              { return s.redirectURL }
func (s *stubProvider) ClientMetadata() OAuthClientMetadata              { return s.clientMeta }
func (s *stubProvider) ClientInformation(ctx context.Context) (*OAuthClientInformation, error) {
	return s.clientInfo, nil
}

func (s *stubProvider) State(ctx context.Context) (string, error)       { return s.state, nil }
func (s *stubProvider) SaveState(ctx context.Context, v string) error   { s.savedState = v; return nil }
func (s *stubProvider) StoredState(ctx context.Context) (string, error) { return s.storedState, nil }
func (s *stubProvider) InvalidateCredentials(ctx context.Context, scope string) error {
	s.invalidations = append(s.invalidations, scope)
	switch scope {
	case "tokens":
		s.tokens = nil
	case "all":
		s.tokens = nil
		s.clientInfo = nil
		s.verifier = ""
	case "verifier":
		s.verifier = ""
	}
	return nil
}
func (s *stubProvider) SaveClientInformation(ctx context.Context, info *OAuthClientInformationFull) error {
	s.savedClientInf = info
	return nil
}

// fakeAuthServer is a minimal in-memory OAuth + protected-resource server
// used to drive Auth() through realistic flows.
func fakeAuthServer(t *testing.T) (string, func()) {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resource":              "http://" + r.Host,
			"authorization_servers": []string{"http://" + r.Host},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthorizationServerMetadata{
			Issuer:                        "http://" + r.Host,
			AuthorizationEndpoint:         "http://" + r.Host + "/auth",
			TokenEndpoint:                 "http://" + r.Host + "/token",
			ResponseTypesSupported:        []string{"code"},
			CodeChallengeMethodsSupported: []string{"S256"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = r.ParseForm()
		switch r.Form.Get("grant_type") {
		case "refresh_token":
			json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "refreshed", TokenType: "Bearer", RefreshToken: "rt2"})
		case "authorization_code":
			json.NewEncoder(w).Encode(OAuthTokens{AccessToken: "exchanged", TokenType: "Bearer", RefreshToken: "rt"})
		default:
			http.Error(w, "bad grant", 400)
		}
	})
	srv := httptest.NewServer(mux)
	return srv.URL, srv.Close
}

func TestAuth_RefreshFlow(t *testing.T) {
	authURL, cleanup := fakeAuthServer(t)
	defer cleanup()

	prov := &stubProvider{
		clientInfo:  &OAuthClientInformation{ClientID: "cid"},
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}},
		redirectURL: "https://app/cb",
		tokens:      &OAuthTokens{AccessToken: "old", RefreshToken: "rt", TokenType: "Bearer"},
	}
	res, err := Auth(context.Background(), prov, AuthOptions{ServerURL: authURL})
	if err != nil {
		t.Fatal(err)
	}
	if res != AuthResultAuthorized {
		t.Errorf("got %s", res)
	}
	if prov.savedTokens == nil || prov.savedTokens.AccessToken != "refreshed" {
		t.Errorf("tokens not saved: %+v", prov.savedTokens)
	}
}

func TestAuth_RedirectFlow_NoExistingTokens(t *testing.T) {
	authURL, cleanup := fakeAuthServer(t)
	defer cleanup()

	prov := &stubProvider{
		clientInfo:  &OAuthClientInformation{ClientID: "cid"},
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}, Scope: "read"},
		redirectURL: "https://app/cb",
	}
	res, err := Auth(context.Background(), prov, AuthOptions{ServerURL: authURL})
	if err != nil {
		t.Fatal(err)
	}
	if res != AuthResultRedirect {
		t.Errorf("got %s", res)
	}
	if prov.redirectedURL == nil {
		t.Fatal("expected RedirectToAuthorization to be called")
	}
	if prov.redirectedURL.Query().Get("client_id") != "cid" {
		t.Errorf("missing client_id in auth URL: %s", prov.redirectedURL)
	}
	if prov.redirectedURL.Query().Get("code_challenge_method") != "S256" {
		t.Errorf("missing PKCE: %s", prov.redirectedURL)
	}
	if prov.savedVerifier == "" {
		t.Error("expected SaveCodeVerifier to be called")
	}
}

func TestAuth_CodeExchange(t *testing.T) {
	authURL, cleanup := fakeAuthServer(t)
	defer cleanup()

	prov := &stubProvider{
		clientInfo:  &OAuthClientInformation{ClientID: "cid"},
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}},
		redirectURL: "https://app/cb",
		verifier:    "stored-verifier",
	}
	res, err := Auth(context.Background(), prov, AuthOptions{ServerURL: authURL, AuthorizationCode: "code-123"})
	if err != nil {
		t.Fatal(err)
	}
	if res != AuthResultAuthorized {
		t.Errorf("got %s", res)
	}
	if prov.savedTokens == nil || prov.savedTokens.AccessToken != "exchanged" {
		t.Errorf("tokens not saved: %+v", prov.savedTokens)
	}
}

func TestAuth_StateMismatch(t *testing.T) {
	authURL, cleanup := fakeAuthServer(t)
	defer cleanup()

	prov := &stubProvider{
		clientInfo:  &OAuthClientInformation{ClientID: "cid"},
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}},
		redirectURL: "https://app/cb",
		verifier:    "v",
		storedState: "expected",
	}
	_, err := Auth(context.Background(), prov, AuthOptions{ServerURL: authURL, AuthorizationCode: "code", CallbackState: "WRONG"})
	if err == nil || !strings.Contains(err.Error(), "state parameter mismatch") {
		t.Errorf("expected CSRF error, got %v", err)
	}
}

func TestAuth_DynamicRegistration(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"resource": "http://" + r.Host, "authorization_servers": []string{"http://" + r.Host}})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthorizationServerMetadata{
			Issuer:                        "http://" + r.Host,
			AuthorizationEndpoint:         "http://" + r.Host + "/auth",
			TokenEndpoint:                 "http://" + r.Host + "/token",
			RegistrationEndpoint:          "http://" + r.Host + "/register",
			ResponseTypesSupported:        []string{"code"},
			CodeChallengeMethodsSupported: []string{"S256"},
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(OAuthClientInformationFull{
			OAuthClientInformation: OAuthClientInformation{ClientID: "dyn-id"},
		})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	prov := &stubProvider{
		clientInfo:  nil, // forces registration
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}},
		redirectURL: "https://app/cb",
	}
	_, err := Auth(context.Background(), prov, AuthOptions{ServerURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if prov.savedClientInf == nil || prov.savedClientInf.ClientID != "dyn-id" {
		t.Errorf("expected dynamic registration to save client info, got %+v", prov.savedClientInf)
	}
}

func TestAuth_InvalidGrantTriggersInvalidate(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"resource": "http://" + r.Host, "authorization_servers": []string{"http://" + r.Host}})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthorizationServerMetadata{
			Issuer:                        "http://" + r.Host,
			AuthorizationEndpoint:         "http://" + r.Host + "/auth",
			TokenEndpoint:                 "http://" + r.Host + "/token",
			ResponseTypesSupported:        []string{"code"},
			CodeChallengeMethodsSupported: []string{"S256"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid_grant"})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	prov := &stubProvider{
		clientInfo:  &OAuthClientInformation{ClientID: "cid"},
		clientMeta:  OAuthClientMetadata{RedirectURIs: []string{"https://app/cb"}},
		redirectURL: "https://app/cb",
		tokens:      &OAuthTokens{AccessToken: "a", RefreshToken: "rt", TokenType: "Bearer"},
	}
	// Expect Auth to invalidate "tokens" on InvalidGrantError, then retry —
	// retry hits /token again with the same 400, so end result is REDIRECT
	// (no tokens, fall through to redirect flow).
	_, err := Auth(context.Background(), prov, AuthOptions{ServerURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if !contains(prov.invalidations, "tokens") {
		t.Errorf("expected tokens invalidation, got %v", prov.invalidations)
	}
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func TestAuth_BubblesNonRecoverableError(t *testing.T) {
	// Auth wraps authInternal; if authInternal returns a non-OAuth error,
	// it should bubble up unchanged.
	prov := &stubProvider{}
	_, err := Auth(context.Background(), prov, AuthOptions{ServerURL: "http://nowhere.invalid"})
	if err == nil {
		t.Fatal("expected error")
	}
	// Just sanity: must not be one of the typed OAuth errors.
	var oauthErr *MCPClientOAuthError
	if errors.As(err, &oauthErr) {
		t.Errorf("unexpected typed OAuth error: %v", err)
	}
}
