package togetherai

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
		if body["top_n"] != float64(1) {
			t.Errorf("body = %v", body)
		}
		w.Write([]byte(`{"id":"rank-1","model":"rerank","results":[{"index":1,"relevance_score":0.9}]}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL}).RerankingModel("rerank").Rerank(context.Background(), model.RerankCallOptions{Query: "query", Documents: []string{"a", "b"}, TopN: 1, ReturnDocuments: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 || result.Results[0].Document != "b" || result.Response.ID != "rank-1" {
		t.Fatalf("result = %+v", result)
	}
}
