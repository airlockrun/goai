package voyage

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestReranking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["top_k"] != float64(1) || body["return_documents"] != true || body["truncation"] != false {
			t.Errorf("body = %v", body)
		}
		w.Write([]byte(`{"data":[{"index":1,"relevance_score":0.9}],"usage":{"total_tokens":4}}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL}).RerankingModel("rerank-2.5").Rerank(context.Background(), model.RerankCallOptions{Query: "query", Documents: []string{"a", "b"}, TopN: 1, ReturnDocuments: true, ProviderOptions: map[string]any{"truncation": false}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || result.Results[0].Document != "b" || result.Results[0].Score != 0.9 || result.Usage.Tokens != 4 {
		t.Fatalf("result = %+v", result)
	}
}
