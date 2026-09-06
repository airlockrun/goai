// Package voyage implements Voyage AI embeddings and reranking.
package voyage

import (
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/stream"
	"net/http"
	"strings"
)

type Options struct {
	APIKey     string
	BaseURL    string
	Headers    map[string]string
	HTTPClient *http.Client
}
type Provider struct{ client *openaicompat.Provider }

func New(opts Options) *Provider {
	if opts.BaseURL == "" {
		opts.BaseURL = "https://api.voyageai.com/v1"
	}
	return &Provider{client: openaicompat.New(openaicompat.Options{ProviderID: "voyage", BaseURL: strings.TrimRight(opts.BaseURL, "/"), APIKey: opts.APIKey, Headers: opts.Headers, HTTPClient: opts.HTTPClient})}
}
func (p *Provider) ID() string                                         { return "voyage" }
func (p *Provider) Model(string) stream.Model                          { return nil }
func (p *Provider) LanguageModel(string) model.LanguageModel           { return nil }
func (p *Provider) ImageModel(string) model.ImageModel                 { return nil }
func (p *Provider) SpeechModel(string) model.SpeechModel               { return nil }
func (p *Provider) TranscriptionModel(string) model.TranscriptionModel { return nil }
func (p *Provider) EmbeddingModel(id string) model.EmbeddingModel {
	return &EmbeddingModel{id: id, provider: p}
}
func (p *Provider) RerankingModel(id string) model.RerankingModel {
	return &RerankingModel{id: id, provider: p}
}

var _ provider.Provider = (*Provider)(nil)
