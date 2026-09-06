package google

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGoogleImageGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-3.5-flash-image:generateContent" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("X-Provider") != "yes" || r.Header.Get("x-goog-api-key") != "key" {
			t.Errorf("headers = %v", r.Header)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		config := request["generationConfig"].(map[string]any)
		if config["seed"] != float64(42) || config["responseModalities"].([]any)[0] != "IMAGE" {
			t.Errorf("config = %v", config)
		}
		image := config["imageConfig"].(map[string]any)
		if image["aspectRatio"] != "16:9" || image["imageSize"] != "2K" {
			t.Errorf("image config = %v", image)
		}
		parts := request["contents"].([]any)[0].(map[string]any)["parts"].([]any)
		if len(parts) != 2 {
			t.Errorf("parts = %v", parts)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ignored"},{"inlineData":{"mimeType":"image/png","data":"dGVzdA=="}}]}}],"usageMetadata":{"totalTokenCount":12}}`))
	}))
	defer server.Close()
	seed := int64(42)
	result, err := New(Options{BaseURL: server.URL, APIKey: "key", Headers: map[string]string{"X-Provider": "yes"}}).ImageModel("gemini-3.5-flash-image").Generate(context.Background(), model.ImageCallOptions{Prompt: "draw", N: 1, Seed: &seed, Size: "1024x1024", AspectRatio: "16:9", Files: [][]byte{[]byte("image")}, ProviderOptions: map[string]any{"imageConfig": map[string]any{"imageSize": "2K"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || result.Images[0].Base64 != "dGVzdA==" || result.Usage.TotalTokens != 12 || len(result.Warnings) != 1 {
		t.Fatalf("result = %+v", result)
	}
}

func TestGoogleImageInvalidOptions(t *testing.T) {
	for _, tc := range []struct {
		name, id string
		options  model.ImageCallOptions
	}{{"retired", "imagen-3", model.ImageCallOptions{}}, {"count", "gemini-image", model.ImageCallOptions{N: 2}}, {"mask", "gemini-image", model.ImageCallOptions{Mask: []byte("mask")}}} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(Options{}).ImageModel(tc.id).Generate(context.Background(), tc.options)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
