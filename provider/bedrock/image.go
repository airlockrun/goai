package bedrock

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/airlockrun/goai/model"
)

// BedrockImageModel implements the ImageModel interface.
type BedrockImageModel struct {
	id       string
	provider *Provider
}

func (m *BedrockImageModel) ID() string            { return m.id }
func (m *BedrockImageModel) Provider() string      { return "bedrock" }
func (m *BedrockImageModel) MaxImagesPerCall() int { return 4 }

func (m *BedrockImageModel) Generate(ctx context.Context, opts model.ImageCallOptions) (*model.ImageResult, error) {
	var reqBody []byte
	var err error

	if strings.HasPrefix(m.id, "amazon.titan-image") {
		reqBody, err = m.buildTitanImageRequest(opts)
	} else if strings.HasPrefix(m.id, "stability.") {
		reqBody, err = m.buildStabilityRequest(opts)
	} else {
		reqBody, err = m.buildTitanImageRequest(opts)
	}

	if err != nil {
		return nil, err
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

	return m.parseImageResponse(body)
}

func (m *BedrockImageModel) buildTitanImageRequest(opts model.ImageCallOptions) ([]byte, error) {
	// Parse size
	width := 1024
	height := 1024
	if opts.Size != "" {
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
	} else if opts.AspectRatio != "" {
		switch opts.AspectRatio {
		case "16:9":
			width, height = 1280, 720
		case "9:16":
			width, height = 720, 1280
		case "4:3":
			width, height = 1024, 768
		case "3:4":
			width, height = 768, 1024
		}
	}

	n := 1
	if opts.N > 0 {
		n = opts.N
	}

	reqBody := map[string]any{
		"textToImageParams": map[string]any{
			"text": opts.Prompt,
		},
		"taskType": "TEXT_IMAGE",
		"imageGenerationConfig": map[string]any{
			"numberOfImages": n,
			"height":         height,
			"width":          width,
			"cfgScale":       8.0,
		},
	}

	if opts.Seed != nil {
		config := reqBody["imageGenerationConfig"].(map[string]any)
		config["seed"] = *opts.Seed
	}

	return json.Marshal(reqBody)
}

func (m *BedrockImageModel) buildStabilityRequest(opts model.ImageCallOptions) ([]byte, error) {
	// Parse size
	width := 1024
	height := 1024
	if opts.Size != "" {
		fmt.Sscanf(opts.Size, "%dx%d", &width, &height)
	}

	reqBody := map[string]any{
		"text_prompts": []map[string]any{
			{"text": opts.Prompt, "weight": 1.0},
		},
		"height": height,
		"width":  width,
		"steps":  50,
	}

	if opts.Seed != nil {
		reqBody["seed"] = *opts.Seed
	}

	return json.Marshal(reqBody)
}

func (m *BedrockImageModel) parseImageResponse(body []byte) (*model.ImageResult, error) {
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	var images []model.GeneratedImage

	// Titan format
	if imgList, ok := resp["images"].([]any); ok {
		for _, img := range imgList {
			if imgStr, ok := img.(string); ok {
				// Decode to verify it's valid base64
				if _, err := base64.StdEncoding.DecodeString(imgStr); err == nil {
					images = append(images, model.GeneratedImage{
						Base64:   imgStr,
						MimeType: "image/png",
					})
				}
			}
		}
	}

	// Stability format
	if artifacts, ok := resp["artifacts"].([]any); ok {
		for _, artifact := range artifacts {
			if art, ok := artifact.(map[string]any); ok {
				if b64, ok := art["base64"].(string); ok {
					images = append(images, model.GeneratedImage{
						Base64:   b64,
						MimeType: "image/png",
					})
				}
			}
		}
	}

	return &model.ImageResult{
		Images: images,
		Response: model.ImageResponse{
			Model: m.id,
		},
	}, nil
}

func (m *BedrockImageModel) signRequest(req *http.Request, payload []byte) {
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

	payloadHash := sha256HashImage(payload)
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
		sha256HashImage([]byte(canonicalRequest)),
	}, "\n")

	kDate := hmacSHA256Image([]byte("AWS4"+m.provider.opts.SecretAccessKey), []byte(dateStamp))
	kRegion := hmacSHA256Image(kDate, []byte(m.provider.opts.Region))
	kService := hmacSHA256Image(kRegion, []byte("bedrock"))
	kSigning := hmacSHA256Image(kService, []byte("aws4_request"))
	signature := hex.EncodeToString(hmacSHA256Image(kSigning, []byte(stringToSign)))

	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, m.provider.opts.AccessKeyID, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)
}

func sha256HashImage(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256Image(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}
