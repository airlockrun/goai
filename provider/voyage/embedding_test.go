package voyage

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" || r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("request = %s %v", r.URL, r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["input_type"] != "query" || body["output_dimension"] != float64(2) || body["truncation"] != false || body["output_dtype"] != "int8" {
			t.Errorf("body = %v", body)
		}
		if _, ok := body["encoding_format"]; ok {
			t.Error("unexpected compatible encoding_format")
		}
		w.Write([]byte(`{"model":"voyage-4","data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"usage":{"total_tokens":12}}`))
	}))
	defer server.Close()
	m := New(Options{BaseURL: server.URL, APIKey: "key"}).EmbeddingModel("voyage-4")
	result, err := m.Embed(context.Background(), model.EmbedCallOptions{Values: []string{"a", "b"}, ProviderOptions: map[string]any{"inputType": "query", "outputDimension": 2, "truncation": false, "outputDtype": "int8"}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Embeddings[0].Values[0] != 1 || result.Usage.Tokens != 12 {
		t.Fatalf("result = %+v", result)
	}
	if _, err := m.Embed(context.Background(), model.EmbedCallOptions{Values: make([]string, 129)}); err == nil {
		t.Fatal("expected batch limit error")
	}
	if _, err := m.Embed(context.Background(), model.EmbedCallOptions{ProviderOptions: map[string]any{"inputType": "invalid"}}); err == nil {
		t.Fatal("expected input type error")
	}
}
