package bedrock

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/airlockrun/goai/model"
)

// BedrockEmbeddingModel implements the EmbeddingModel interface.
type BedrockEmbeddingModel struct {
	id       string
	provider *Provider
}

func (m *BedrockEmbeddingModel) ID() string                { return m.id }
func (m *BedrockEmbeddingModel) Provider() string          { return "bedrock" }
func (m *BedrockEmbeddingModel) MaxEmbeddingsPerCall() int { return 1 } // Bedrock does one at a time
func (m *BedrockEmbeddingModel) Dimensions() int           { return 0 } // Variable

func (m *BedrockEmbeddingModel) Embed(ctx context.Context, opts model.EmbedCallOptions) (*model.EmbedResult, error) {
	embeddings := make([]model.Embedding, len(opts.Values))
	totalTokens := 0

	for i, text := range opts.Values {
		var reqBody []byte
		var err error

		if strings.HasPrefix(m.id, "amazon.titan-embed") {
			reqBody, err = json.Marshal(map[string]any{
				"inputText": text,
			})
		} else if strings.HasPrefix(m.id, "cohere.embed") {
			reqBody, err = json.Marshal(map[string]any{
				"texts":      []string{text},
				"input_type": "search_document",
			})
		} else {
			// Default to Titan format
			reqBody, err = json.Marshal(map[string]any{
				"inputText": text,
			})
		}

		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		url := fmt.Sprintf("%s/model/%s/invoke", m.provider.baseURL(), m.id)

		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Content-Type", "application/json")
		m.signRequest(req, reqBody)

		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request failed: %w", err)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("Bedrock API error (status %d): %s", resp.StatusCode, string(body))
		}

		var embResp map[string]any
		if err := json.Unmarshal(body, &embResp); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}

		// Extract embedding based on model
		var embedding []float64
		if emb, ok := embResp["embedding"].([]any); ok {
			embedding = make([]float64, len(emb))
			for j, v := range emb {
				if f, ok := v.(float64); ok {
					embedding[j] = f
				}
			}
		} else if embs, ok := embResp["embeddings"].([]any); ok && len(embs) > 0 {
			// Cohere format
			if emb, ok := embs[0].([]any); ok {
				embedding = make([]float64, len(emb))
				for j, v := range emb {
					if f, ok := v.(float64); ok {
						embedding[j] = f
					}
				}
			}
		}

		embeddings[i] = model.Embedding{
			Values: embedding,
			Index:  i,
		}

		// Track tokens if available
		if tokens, ok := embResp["inputTextTokenCount"].(float64); ok {
			totalTokens += int(tokens)
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: model.EmbeddingUsage{
			Tokens: totalTokens,
		},
		Response: model.EmbeddingResponse{
			Model: m.id,
		},
	}, nil
}

func (m *BedrockEmbeddingModel) signRequest(req *http.Request, payload []byte) {
	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("X-Amz-Date", amzDate)
	if m.provider.opts.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", m.provider.opts.SessionToken)
	}

	signedHeaders := "content-type;host;x-amz-date"
	if m.provider.opts.SessionToken != "" {
		signedHeaders = "content-type;host;x-amz-date;x-amz-security-token"
	}

	payloadHash := sha256HashEmbed(payload)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	canonicalHeaders := fmt.Sprintf("content-type:%s\nhost:%s\nx-amz-date:%s\n",
		req.Header.Get("Content-Type"), req.Host, amzDate)
	if m.provider.opts.SessionToken != "" {
		canonicalHeaders = fmt.Sprintf("content-type:%s\nhost:%s\nx-amz-date:%s\nx-amz-security-token:%s\n",
			req.Header.Get("Content-Type"), req.Host, amzDate, m.provider.opts.SessionToken)
	}

	canonicalRequest := strings.Join([]string{
		req.Method,
		req.URL.Path,
		req.URL.RawQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	algorithm := "AWS4-HMAC-SHA256"
	credentialScope := fmt.Sprintf("%s/%s/bedrock/aws4_request", dateStamp, m.provider.opts.Region)
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		credentialScope,
		sha256HashEmbed([]byte(canonicalRequest)),
	}, "\n")

	kDate := hmacSHA256Embed([]byte("AWS4"+m.provider.opts.SecretAccessKey), []byte(dateStamp))
	kRegion := hmacSHA256Embed(kDate, []byte(m.provider.opts.Region))
	kService := hmacSHA256Embed(kRegion, []byte("bedrock"))
	kSigning := hmacSHA256Embed(kService, []byte("aws4_request"))
	signature := hex.EncodeToString(hmacSHA256Embed(kSigning, []byte(stringToSign)))

	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, m.provider.opts.AccessKeyID, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)
}

func sha256HashEmbed(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256Embed(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}
