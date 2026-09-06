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

func TestReranking(t *testing.T) {
	original := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = original })
	for _, tc := range []struct {
		name, response string
		wantErr        bool
	}{{"success", `{"results":[{"index":1,"relevanceScore":0.9}]}`, false}, {"bad index", `{"results":[{"index":2,"relevanceScore":0.9}]}`, true}, {"missing results", `{}`, true}} {
		t.Run(tc.name, func(t *testing.T) {
			http.DefaultClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if req.URL.Host != "bedrock-agent-runtime.us-east-1.amazonaws.com" || req.URL.Path != "/rerank" || req.Header.Get("Authorization") == "" {
					t.Fatalf("%v", req)
				}
				var body map[string]any
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				config := body["rerankingConfiguration"].(map[string]any)["bedrockRerankingConfiguration"].(map[string]any)
				if config["numberOfResults"] != float64(1) || config["modelConfiguration"].(map[string]any)["modelArn"] != "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0" {
					t.Fatalf("%v", body)
				}
				return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(tc.response))}, nil
			})}
			result, err := New(Options{AccessKeyID: "key", SecretAccessKey: "secret"}).RerankingModel("amazon.rerank-v1:0").Rerank(context.Background(), model.RerankCallOptions{Query: "fruit", Documents: []string{"car", "apple"}, TopN: 1, ReturnDocuments: true})
			if (err != nil) != tc.wantErr {
				t.Fatalf("%v", err)
			}
			if err == nil && result.Results[0].Document != "apple" {
				t.Fatalf("%+v", result)
			}
		})
	}
}
