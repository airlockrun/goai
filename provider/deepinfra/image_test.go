package deepinfra

import (
	"context"
	"encoding/json"
	"github.com/airlockrun/goai/model"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestImageModes(t *testing.T) {
	for _, editing := range []bool{false, true} {
		name := "generate"
		if editing {
			name = "edit"
		}
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if editing {
					if r.URL.Path != "/v1/openai/images/edits" {
						t.Errorf("path = %s", r.URL.Path)
					}
					if err := r.ParseMultipartForm(1 << 20); err != nil {
						t.Error(err)
						return
					}
					defer r.MultipartForm.RemoveAll()
					if len(r.MultipartForm.File["image"]) != 2 || len(r.MultipartForm.File["mask"]) != 1 {
						t.Errorf("files = %v", r.MultipartForm.File)
					}
					w.Write([]byte(`{"data":[{"b64_json":"aW1hZ2U="}]}`))
					return
				}
				if r.URL.Path != "/v1/inference/flux" {
					t.Errorf("path = %s", r.URL.Path)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if body["width"] != float64(512) || body["num_images"] != float64(1) {
					t.Errorf("body = %v", body)
				}
				w.Write([]byte(`{"images":["data:image/png;base64,aW1hZ2U="]}`))
			}))
			defer server.Close()
			opts := model.ImageCallOptions{Prompt: "hello", Size: "512x512"}
			if editing {
				opts.Files = [][]byte{[]byte("a"), []byte("b")}
				opts.Mask = []byte("mask")
			}
			result, err := New(Options{BaseURL: server.URL + "/v1/openai"}).ImageModel("flux").Generate(context.Background(), opts)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Images) != 1 || result.Images[0].Base64 != "aW1hZ2U=" {
				t.Fatalf("result = %+v", result)
			}
		})
	}
}
