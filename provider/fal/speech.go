package fal

import (
	"context"
	"errors"
	"fmt"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
)

type FalSpeechModel struct {
	id       string
	provider *Provider
}

func (m *FalSpeechModel) ID() string       { return m.id }
func (m *FalSpeechModel) Provider() string { return "fal" }
func (m *FalSpeechModel) Generate(ctx context.Context, opts model.SpeechCallOptions) (*model.SpeechResult, error) {
	baseURL := m.provider.opts.BaseURL
	if baseURL == defaultBaseURL {
		baseURL = "https://fal.run"
	}
	body := map[string]any{"text": opts.Text, "output_format": "url"}
	if opts.Voice != "" {
		body["voice"] = opts.Voice
	}
	if opts.Speed != nil {
		body["speed"] = *opts.Speed
	}
	for k, v := range opts.ProviderOptions {
		body[k] = v
	}
	var warnings []stream.Warning
	if opts.OutputFormat != "" && opts.OutputFormat != "url" {
		warnings = append(warnings, stream.UnsupportedWarning("outputFormat", "using url"))
	}
	client := openaicompat.New(openaicompat.Options{ProviderID: m.Provider(), BaseURL: strings.TrimRight(baseURL, "/"), APIKey: m.provider.opts.APIKey, Headers: m.provider.opts.Headers, AuthPrefix: "Key "})
	var result struct {
		Audio struct {
			URL string `json:"url"`
		} `json:"audio"`
		Duration *float64 `json:"duration_ms"`
		ID       string   `json:"request_id"`
	}
	headers, err := client.DoJSON(ctx, "/"+m.id, body, &result, opts.Headers)
	if err != nil {
		return nil, err
	}
	u, err := url.Parse(result.Audio.URL)
	if err != nil {
		return nil, err
	}
	trusted, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	validate := func(u *url.URL) error {
		if u.User != nil || u.Hostname() == "" || (u.Scheme != "https" && !(u.Scheme == trusted.Scheme && u.Host == trusted.Host)) {
			return errors.New("unsafe fal audio URL")
		}
		return nil
	}
	if err := validate(u); err != nil {
		return nil, err
	}
	trustedPort := trusted.Port()
	if trustedPort == "" {
		trustedPort = "443"
		if trusted.Scheme == "http" {
			trustedPort = "80"
		}
	}
	// Resolve and dial the validated address together to prevent DNS rebinding.
	transport := &http.Transport{DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		for _, ip := range ips {
			if (host != trusted.Hostname() || port != trustedPort) && (!ip.IP.IsGlobalUnicast() || ip.IP.IsPrivate() || ip.IP.IsLoopback() || ip.IP.IsLinkLocalUnicast()) {
				return nil, errors.New("unsafe fal audio address")
			}
		}
		var dialer net.Dialer
		for _, ip := range ips {
			conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(ip.IP.String(), port))
			if err == nil {
				return conn, nil
			}
		}
		return nil, errors.New("cannot connect to fal audio host")
	}}
	defer transport.CloseIdleConnections()
	downloader := &http.Client{Transport: transport, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("too many audio redirects")
		}
		return validate(req.URL)
	}}
	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := downloader.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fal audio download status %d", resp.StatusCode)
	}
	audio, err := io.ReadAll(io.LimitReader(resp.Body, (100<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(audio) > 100<<20 {
		return nil, errors.New("fal audio exceeds 100 MiB")
	}
	var duration *float64
	if result.Duration != nil {
		seconds := *result.Duration / 1000
		duration = &seconds
	}
	return &model.SpeechResult{Audio: audio, MimeType: resp.Header.Get("Content-Type"), Duration: duration, Warnings: warnings, Response: model.SpeechResponse{ID: result.ID, Model: m.id, Headers: headers}}, nil
}
