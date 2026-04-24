// Package vertexmaas provides a Google Vertex AI MaaS (Model as a Service)
// provider. It wraps the openai-compatible provider base, pointing at Vertex's
// /endpoints/openapi proxy for partner/open models. Mirrors ai-sdk's
// packages/google-vertex/src/maas/google-vertex-maas-provider.ts.
package vertexmaas

import (
	"fmt"
	"maps"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
)

// VertexMaasModels enumerates the MaaS models published through Vertex AI's
// OpenAI-compatible endpoint. Mirrors ai-sdk's GoogleVertexMaasModelId union
// at packages/google-vertex/src/maas/google-vertex-maas-options.ts.
var VertexMaasModels = []string{
	"deepseek-ai/deepseek-r1-0528-maas",
	"deepseek-ai/deepseek-v3.1-maas",
	"deepseek-ai/deepseek-v3.2-maas",
	"openai/gpt-oss-120b-maas",
	"openai/gpt-oss-20b-maas",
	"meta/llama-4-maverick-17b-128e-instruct-maas",
	"meta/llama-4-scout-17b-16e-instruct-maas",
	"minimax/minimax-m2-maas",
	"qwen/qwen3-coder-480b-a35b-instruct-maas",
	"qwen/qwen3-next-80b-a3b-instruct-maas",
	"qwen/qwen3-next-80b-a3b-thinking-maas",
	"moonshotai/kimi-k2-thinking-maas",
}

// Options contains configuration for the Vertex MaaS provider. Callers supply
// the Google Cloud project + location and an OAuth bearer token directly;
// goai never reads environment variables.
type Options struct {
	// Project is the Google Cloud project ID.
	Project string

	// Location is the Google Cloud region. Defaults to "global" (the MaaS
	// endpoint most models land on).
	Location string

	// AccessToken is the OAuth bearer token sent as Authorization:
	// Bearer {AccessToken}.
	AccessToken string

	// BaseURL overrides the computed Vertex endpoint.
	BaseURL string

	// Headers are additional HTTP headers to send.
	Headers map[string]string
}

// Provider implements the Vertex MaaS provider.
type Provider struct {
	opts   Options
	compat *openaicompat.Provider
}

// New creates a new Vertex MaaS provider.
func New(opts Options) *Provider {
	if opts.Location == "" {
		opts.Location = "global"
	}
	baseURL := strings.TrimRight(opts.BaseURL, "/")
	if baseURL == "" {
		baseURL = fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/%s/endpoints/openapi",
			opts.Project, opts.Location,
		)
	}
	headers := maps.Clone(opts.Headers)
	if headers == nil {
		headers = map[string]string{}
	}
	headers["Authorization"] = "Bearer " + opts.AccessToken

	return &Provider{
		opts: opts,
		compat: openaicompat.New(openaicompat.Options{
			ProviderID: "vertex.maas",
			BaseURL:    baseURL,
			Headers:    headers,
			// AuthScheme values are irrelevant when APIKey is unset; Bearer is
			// already on Headers.
		}),
	}
}

// ID returns the provider identifier.
func (p *Provider) ID() string { return "vertex.maas" }

// LanguageModel returns a language model instance for the given model ID.
func (p *Provider) LanguageModel(modelID string) model.LanguageModel {
	return p.compat.Model(modelID)
}

// Model returns the generic Model interface for modelID.
func (p *Provider) Model(modelID string) stream.Model {
	return p.compat.Model(modelID)
}

// ImageModel returns nil; MaaS is chat-completions only.
func (p *Provider) ImageModel(modelID string) model.ImageModel { return nil }

// EmbeddingModel returns nil.
func (p *Provider) EmbeddingModel(modelID string) model.EmbeddingModel { return nil }

// SpeechModel returns nil.
func (p *Provider) SpeechModel(modelID string) model.SpeechModel { return nil }

// TranscriptionModel returns nil.
func (p *Provider) TranscriptionModel(modelID string) model.TranscriptionModel { return nil }

// RerankingModel returns nil.
func (p *Provider) RerankingModel(modelID string) model.RerankingModel { return nil }

// Models returns the Vertex MaaS model catalog.
func (p *Provider) Models() []string {
	out := make([]string, len(VertexMaasModels))
	copy(out, VertexMaasModels)
	return out
}

var _ provider.Provider = (*Provider)(nil)
