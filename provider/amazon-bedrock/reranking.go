package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/response"
)

type BedrockRerankingModel struct {
	id       string
	provider *Provider
}

func (m *BedrockRerankingModel) ID() string               { return m.id }
func (m *BedrockRerankingModel) Provider() string         { return "amazon-bedrock" }
func (m *BedrockRerankingModel) MaxDocumentsPerCall() int { return 1000 }
func (m *BedrockRerankingModel) Rerank(ctx context.Context, opts model.RerankCallOptions) (*model.RerankResult, error) {
	if len(opts.Documents) == 0 || len(opts.Documents) > m.MaxDocumentsPerCall() || opts.TopN < 0 || opts.TopN > len(opts.Documents) {
		return nil, errors.New("invalid Bedrock reranking document count or topN")
	}
	arn := m.id
	if !strings.HasPrefix(arn, "arn:") {
		partition := "aws"
		if strings.HasPrefix(m.provider.opts.Region, "cn-") {
			partition = "aws-cn"
		}
		if strings.HasPrefix(m.provider.opts.Region, "us-gov-") {
			partition = "aws-us-gov"
		}
		arn = fmt.Sprintf("arn:%s:bedrock:%s::foundation-model/%s", partition, m.provider.opts.Region, m.id)
	}
	sources := make([]any, len(opts.Documents))
	for i, doc := range opts.Documents {
		sources[i] = map[string]any{"type": "INLINE", "inlineDocumentSource": map[string]any{"type": "TEXT", "textDocument": map[string]any{"text": doc}}}
	}
	n := opts.TopN
	if n == 0 {
		n = len(sources)
	}
	config := map[string]any{"modelArn": arn}
	if len(opts.ProviderOptions) > 0 {
		config["additionalModelRequestFields"] = opts.ProviderOptions
	}
	body, err := json.Marshal(map[string]any{"queries": []any{map[string]any{"type": "TEXT", "textQuery": map[string]any{"text": opts.Query}}}, "sources": sources, "rerankingConfiguration": map[string]any{"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": map[string]any{"modelConfiguration": config, "numberOfResults": n}}})
	if err != nil {
		return nil, err
	}
	endpoint := fmt.Sprintf("https://bedrock-agent-runtime.%s.amazonaws.com/rerank", m.provider.opts.Region)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.provider.opts.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}
	(&BedrockLanguageModel{provider: m.provider}).signRequest(req, body)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: "Bedrock reranking request failed", URL: endpoint, Cause: err, IsRetryable: ctx.Err() == nil, IsRetryableSet: true})
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: "Bedrock reranking failed", URL: endpoint, StatusCode: resp.StatusCode, ResponseBody: string(raw), ResponseHeaders: response.ExtractResponseHeaders(resp)})
	}
	var result struct {
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevanceScore"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	if result.Results == nil {
		return nil, errors.New("Bedrock reranking response missing results")
	}
	out := &model.RerankResult{Response: model.RerankResponse{Model: m.id, Headers: response.ExtractResponseHeaders(resp)}}
	for _, r := range result.Results {
		if r.Index < 0 || r.Index >= len(opts.Documents) {
			return nil, errors.New("Bedrock reranking result index out of bounds")
		}
		doc := model.RankedDocument{Index: r.Index, Score: r.Score}
		if opts.ReturnDocuments {
			doc.Document = opts.Documents[r.Index]
		}
		out.Results = append(out.Results, doc)
	}
	return out, nil
}
