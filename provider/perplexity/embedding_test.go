package perplexity

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbeddingQuantization(t *testing.T) {
	for _, tc := range []struct {
		format string
		want   float64
	}{{"base64_int8", -1}, {"base64_binary", 255}} {
		t.Run(tc.format, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/embeddings" {
					t.Errorf("path = %s", r.URL.Path)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body["encoding_format"] != tc.format {
					t.Errorf("body = %v", body)
				}
				w.Write([]byte(`{"data":[{"embedding":"/wB/"}],"usage":{"prompt_tokens":3}}`))
			}))
			defer server.Close()
			result, err := New(Options{BaseURL: server.URL}).EmbeddingModel("pplx-embed-v1-0.6b").Embed(context.Background(), model.EmbedCallOptions{Values: []string{"a"}, ProviderOptions: map[string]any{"encodingFormat": tc.format}})
			if err != nil {
				t.Fatal(err)
			}
			if result.Embeddings[0].Values[0] != tc.want || result.Usage.Tokens != 3 {
				t.Fatalf("result = %+v", result)
			}
		})
	}
}
