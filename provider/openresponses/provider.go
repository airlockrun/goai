// Package openresponses provides a configurable Responses API provider.
package openresponses

import (
	"net/http"
	"strings"

	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/openai"
	"github.com/airlockrun/goai/stream"
)

// Options configures a Responses-compatible endpoint. BaseURL is required.
type Options struct {
	Name       string
	BaseURL    string
	APIKey     string
	Headers    map[string]string
	HTTPClient *http.Client
}

// ResponsesOptions configures Responses requests.
type ResponsesOptions = openai.ResponsesOptions

type Provider struct{ opts Options }

func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		panic("openresponses: BaseURL is required")
	}
	if opts.Name == "" {
		opts.Name = "openresponses"
	}
	return &Provider{opts: opts}
}

func (p *Provider) ID() string                                  { return p.opts.Name }
func (p *Provider) Model(id string) stream.Model                { return p.Responses(id) }
func (p *Provider) LanguageModel(id string) model.LanguageModel { return p.Responses(id) }
func (p *Provider) Responses(id string) *openai.ResponsesModel {
	return openai.NewResponsesModel(id, openai.ResponsesConfig{
		Provider: p.ID() + ".responses", URL: strings.TrimRight(p.opts.BaseURL, "/") + "/responses",
		HTTPClient: p.opts.HTTPClient, Generic: true, Headers: p.opts.Headers,
		ConfigureRequest: func(req *http.Request) error {
			if p.opts.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+p.opts.APIKey)
			}
			return nil
		},
	})
}

func (p *Provider) ImageModel(string) model.ImageModel                 { return nil }
func (p *Provider) EmbeddingModel(string) model.EmbeddingModel         { return nil }
func (p *Provider) SpeechModel(string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(string) model.TranscriptionModel { return nil }
func (p *Provider) RerankingModel(string) model.RerankingModel         { return nil }
