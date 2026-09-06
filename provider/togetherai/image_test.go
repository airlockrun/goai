package togetherai

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestImageGeneration(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["width"] != float64(512) || body["height"] != float64(256) || body["response_format"] != "base64" || !strings.HasPrefix(body["image_url"].(string), "data:") {
			t.Errorf("body = %v", body)
		}
		w.Write([]byte(`{"data":[{"b64_json":"aW1hZ2U="}]}`))
	}))
	defer server.Close()
	result, err := New(Options{BaseURL: server.URL}).ImageModel("flux").Generate(context.Background(), model.ImageCallOptions{Prompt: "hello", Size: "512x256", Files: [][]byte{[]byte("source")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || result.Images[0].Base64 != "aW1hZ2U=" {
		t.Fatalf("result = %+v", result)
	}
}
