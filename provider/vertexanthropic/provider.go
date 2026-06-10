// Package vertexanthropic provides a Google Vertex AI Anthropic provider.
// It wires Vertex-specific hooks into the shared Anthropic request builder
// exposed by goai/provider/anthropic. Mirrors ai-sdk's
// packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts.
package vertexanthropic

import (
	"fmt"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/stream"
)

// Options contains configuration for the Vertex Anthropic provider. Mirrors
// ai-sdk's GoogleVertexAnthropicProviderSettings. Callers supply the
// Google Cloud project + location and an OAuth bearer token directly;
// goai never reads environment variables.
type Options struct {
	// Project is the Google Cloud project ID.
	Project string

	// Location is the Google Cloud region (e.g. "us-east5"). Use "global"
	// for the region-less endpoint. Defaults to "us-central1" when empty.
	Location string

	// AccessToken is the OAuth bearer token sent as Authorization:
	// Bearer {AccessToken}.
	AccessToken string

	// BaseURL overrides the computed Vertex endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Vertex Anthropic provider.
type Provider struct {
	opts Options
}

// New creates a new Vertex Anthropic provider.
func New(opts Options) *Provider {
	if opts.Location == "" {
		opts.Location = "us-central1"
	}
	return &Provider{opts: opts}
}

// ID returns the provider identifier.
func (p *Provider) ID() string { return "vertex.anthropic" }

// baseURL returns the Vertex Anthropic endpoint for this provider
// instance. Mirrors ai-sdk's getBaseURL:
//
//	https://{host}/v1/projects/{project}/locations/{location}/publishers/anthropic/models
//
// where {host} is derived from Location by vertexHost.
func (p *Provider) baseURL() string {
	if p.opts.BaseURL != "" {
		return strings.TrimRight(p.opts.BaseURL, "/")
	}
	return fmt.Sprintf(
		"https://%s/v1/projects/%s/locations/%s/publishers/anthropic/models",
		vertexHost(p.opts.Location), p.opts.Project, p.opts.Location,
	)
}

// vertexHost maps a Vertex location to its API host. "global" uses the
// region-less host, the "eu" and "us" multi-region locations use the
// dedicated `.rep.` hosts, and every other location is region-prefixed.
// Mirrors ai-sdk's getHost (#bb93832).
func vertexHost(location string) string {
	switch location {
	case "global":
		return "aiplatform.googleapis.com"
	case "eu", "us":
		return fmt.Sprintf("aiplatform.%s.rep.googleapis.com", location)
	default:
		return fmt.Sprintf("%s-aiplatform.googleapis.com", location)
	}
}

// config builds the per-call anthropic.Config with Vertex hooks. Captured
// modelID is embedded into BuildRequestURL because Vertex constructs the
// endpoint path as {baseURL}/{model}:rawPredict (or :streamRawPredict).
func (p *Provider) config(modelID string) anthropic.Config {
	falseVal := false
	return anthropic.Config{
		ProviderID: "vertex.anthropic.messages",

		BuildRequestURL: func(baseURL string, streaming bool) string {
			suffix := "rawPredict"
			if streaming {
				suffix = "streamRawPredict"
			}
			return baseURL + "/" + modelID + ":" + suffix
		},

		TransformRequestBody: func(body map[string]any, betas []string) map[string]any {
			// Vertex derives the model from the URL path and expects the
			// anthropic_version sentinel in the body.
			delete(body, "model")
			body["anthropic_version"] = "vertex-2023-10-16"
			return body
		},

		// Vertex Anthropic doesn't support the beta header for native
		// structured outputs or strict tool definitions.
		SupportsNativeStructuredOutput: &falseVal,
		SupportsStrictTools:            &falseVal,

		// Empty allowlist forces the core to download + base64 every URL.
		SupportedURLs: func() map[string][]string {
			return map[string][]string{}
		},

		// EmitBetasInBody stays false: Vertex accepts the anthropic-beta
		// header, same as direct Anthropic.
	}
}

// LanguageModel returns a language model instance backed by the shared
// Anthropic provider configured with Vertex hooks.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	inner := anthropic.NewWithConfig(anthropic.Options{
		APIKey:     p.opts.AccessToken,
		BaseURL:    p.baseURL(),
		Headers:    p.opts.Headers,
		AuthScheme: "bearer",
	}, p.config(modelID))
	return inner.LanguageModel(modelID)
}

// Model returns the generic Model interface for modelID.
func (p *Provider) Model(modelID string) stream.Model {
	return p.LanguageModel(modelID)
}

// ImageModel returns nil; Vertex Anthropic has no image models.
func (p *Provider) ImageModel(modelID string) model.ImageModel { return nil }

// EmbeddingModel returns nil; Vertex Anthropic has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }

// SpeechModel returns nil.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel { return nil }

// TranscriptionModel returns nil.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }

// RerankingModel returns nil.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

// Models returns the Vertex Anthropic chat-model catalog.
func (p *Provider) Models() []string {
	out := make([]string, len(VertexAnthropicChatModels))
	copy(out, VertexAnthropicChatModels)
	return out
}

var _ provider.Provider = (*Provider)(nil)
