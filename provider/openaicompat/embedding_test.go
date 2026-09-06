package openaicompat_test

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider/baseten"
	"github.com/airlockrun/goai/provider/fireworks"
	"github.com/airlockrun/goai/provider/togetherai"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompatibleEmbeddingProviders(t *testing.T) {
	for _, tc := range []struct {
		name, path string
		create     func(string) model.EmbeddingModel
	}{
		{"fireworks", "/embeddings", func(url string) model.EmbeddingModel {
			return fireworks.New(fireworks.Options{BaseURL: url, APIKey: "key"}).EmbeddingModel("embed")
		}},
		{"together", "/embeddings", func(url string) model.EmbeddingModel {
			return togetherai.New(togetherai.Options{BaseURL: url, APIKey: "key"}).EmbeddingModel("embed")
		}},
		{"baseten", "/sync/v1/embeddings", func(url string) model.EmbeddingModel {
			return baseten.New(baseten.Options{ModelURL: url + "/sync", APIKey: "key"}).EmbeddingModel("embed")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.path || r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("X-Call") != "yes" {
					t.Errorf("request = %s %v", r.URL, r.Header)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body["encoding_format"] != "float" || body["dimensions"] != float64(2) {
					t.Errorf("body = %v", body)
				}
				w.Write([]byte(`{"model":"embed","data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"usage":{"total_tokens":7}}`))
			}))
			defer server.Close()
			dimensions := 2
			result, err := tc.create(server.URL).Embed(context.Background(), model.EmbedCallOptions{Values: []string{"a", "b"}, Dimensions: &dimensions, Headers: map[string]string{"X-Call": "yes"}})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Embeddings) != 2 || result.Embeddings[0].Values[0] != 1 || result.Usage.Tokens != 7 {
				t.Fatalf("result = %+v", result)
			}
		})
	}
}
