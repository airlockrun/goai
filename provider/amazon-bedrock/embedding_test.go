package bedrock

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/airlockrun/goai/model"
)

func TestEmbeddingFamilies(t *testing.T) {
	original := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = original })
	for _, tc := range []struct{ id, key, response string }{
		{"amazon.titan-embed-text-v2:0", "inputText", `{"embedding":[0.1,0.2],"inputTextTokenCount":2}`},
		{"global.cohere.embed-v4:0", "texts", `{"embeddings":{"float":[[0.1,0.2]]}}`},
		{"arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3", "texts", `{"embeddings":[[0.1,0.2]]}`},
		{"us.amazon.nova-2-multimodal-embeddings-v1:0", "taskType", `{"embeddings":[{"embeddingType":"TEXT","embedding":[0.1,0.2]}],"inputTokenCount":2}`},
	} {
		t.Run(tc.id, func(t *testing.T) {
			http.DefaultClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				var body map[string]any
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if body[tc.key] == nil {
					t.Fatalf("%v", body)
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(tc.response))}, nil
			})}
			result, err := New(Options{}).EmbeddingModel(tc.id).Embed(context.Background(), model.EmbedCallOptions{Values: []string{"hello"}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Embeddings) != 1 || len(result.Embeddings[0].Values) != 2 || result.Embeddings[0].Values[0] != 0.1 {
				t.Fatalf("%+v", result)
			}
		})
	}
	if _, err := New(Options{}).EmbeddingModel("unknown").Embed(context.Background(), model.EmbedCallOptions{Values: []string{"hi"}}); err == nil {
		t.Fatal("unknown model accepted")
	}
}
