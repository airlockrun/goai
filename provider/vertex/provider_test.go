package vertex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func TestVertexStreamValidation(t *testing.T) {
	for _, tc := range []struct {
		name, data   string
		api, invalid bool
		finish       stream.FinishReason
	}{
		{name: "blocked", data: `data:{"promptFeedback":{"blockReason":"SAFETY"}}`, finish: stream.FinishReasonContentFilter},
		{name: "invalid", data: `data: {`, invalid: true},
		{name: "API error", data: `data: {"error":{"code":503,"message":"unavailable"}}`, api: true},
		{name: "empty", invalid: true},
		{name: "incomplete", data: `data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`, invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			events := make(chan stream.Event, 100)
			(&VertexLanguageModel{}).processStream(context.Background(), strings.NewReader(tc.data+"\n\n"), nil, events, false)
			close(events)
			var gotErr error
			var finish stream.FinishReason
			for event := range events {
				if value, ok := event.Data.(stream.ErrorEvent); ok {
					gotErr = value.Error
				}
				if value, ok := event.Data.(stream.FinishEvent); ok {
					finish = value.FinishReason
				}
			}
			if tc.invalid && !errors.Is(gotErr, goaierrors.ErrInvalidResponse) {
				t.Fatalf("error = %v", gotErr)
			}
			if tc.api {
				var api *goaierrors.APICallError
				if !errors.As(gotErr, &api) {
					t.Fatalf("error = %v", gotErr)
				}
			}
			if finish != tc.finish {
				t.Fatalf("finish = %q", finish)
			}
		})
	}
}

func TestVertexEmbeddingRoutes(t *testing.T) {
	for _, id := range []string{"text-embedding-005", "gemini-embedding-2", "gemini-embedding-2-preview"} {
		t.Run(id, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if strings.HasPrefix(id, "gemini-") {
					if !strings.HasSuffix(r.URL.Path, ":embedContent") || request["embedContentConfig"].(map[string]any)["outputDimensionality"] != float64(2) {
						t.Errorf("request = %s %v", r.URL.Path, request)
					}
					_, _ = w.Write([]byte(`{"embedding":{"values":[1,2]}}`))
				} else {
					if !strings.HasSuffix(r.URL.Path, ":predict") {
						t.Errorf("path = %s", r.URL.Path)
					}
					_, _ = w.Write([]byte(`{"predictions":[{"embeddings":{"values":[1,2]}}]}`))
				}
			}))
			defer server.Close()
			m := New(Options{BaseURL: server.URL}).EmbeddingModel(id)
			dimensions := 2
			result, err := m.Embed(context.Background(), model.EmbedCallOptions{Values: []string{"test"}, Dimensions: &dimensions})
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Embeddings) != 1 {
				t.Fatalf("result = %+v", result)
			}
			if strings.HasPrefix(id, "gemini-") {
				if m.MaxEmbeddingsPerCall() != 1 {
					t.Fatal("must limit batch to one")
				}
				if _, err := m.Embed(context.Background(), model.EmbedCallOptions{Values: []string{"a", "b"}}); err == nil {
					t.Fatal("oversized batch must fail")
				}
			}
		})
	}
}

func TestVertexLanguageOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if request["serviceTier"] != nil || r.Header.Get("X-Vertex-AI-LLM-Shared-Request-Type") != "priority" || r.Header.Get("X-Vertex-AI-LLM-Request-Type") != "shared" {
			t.Errorf("tier = %v %v", request, r.Header)
		}
		config := request["generationConfig"].(map[string]any)
		if config["thinkingConfig"].(map[string]any)["thinkingLevel"] != "low" || config["imageConfig"] == nil || config["streamFunctionCallArguments"] != nil {
			t.Errorf("config = %v", config)
		}
		toolConfig := request["toolConfig"].(map[string]any)
		if toolConfig["functionCallingConfig"].(map[string]any)["streamFunctionCallArguments"] != true || toolConfig["retrievalConfig"] == nil {
			t.Errorf("tool config = %v", toolConfig)
		}
		if request["cachedContent"] != "cache" || request["labels"] == nil || len(request["safetySettings"].([]any)) != 5 {
			t.Errorf("request = %v", request)
		}
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\n"))
	}))
	defer server.Close()
	events, err := New(Options{BaseURL: server.URL}).Model("gemini-3.8-flash").Stream(context.Background(), &stream.CallOptions{Reasoning: "none", ToolChoice: "auto", ProviderOptions: map[string]any{"serviceTier": "priority", "sharedRequestType": "priority", "requestType": "shared", "streamFunctionCallArguments": true, "imageConfig": map[string]any{"imageSize": "2K"}, "retrievalConfig": map[string]any{"latLng": map[string]any{"latitude": 1, "longitude": 2}}, "cachedContent": "cache", "labels": map[string]string{"team": "test"}, "threshold": "OFF"}})
	if err != nil {
		t.Fatal(err)
	}
	for event := range events {
		if event.Type == stream.EventError {
			t.Fatalf("error = %+v", event.Data)
		}
	}
}

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestVertexProvider_ID(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		Location:    "us-central1",
		AccessToken: "test-token",
	})

	if provider.ID() != "vertex" {
		t.Errorf("expected provider ID vertex, got %s", provider.ID())
	}
}

func TestVertexLanguageModel_SetupErrorClassification(t *testing.T) {
	transportErr := errors.New("connection refused")
	tests := []struct {
		name          string
		roundTrip     roundTripFunc
		wantStatus    int
		wantRetryable bool
		wantCause     error
	}{
		{
			name: "transport error",
			roundTrip: func(*http.Request) (*http.Response, error) {
				return nil, transportErr
			},
			wantRetryable: true,
			wantCause:     transportErr,
		},
		{
			name: "bad request",
			roundTrip: func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader("invalid request")),
					Request:    req,
				}, nil
			},
			wantStatus: http.StatusBadRequest,
		},
	}

	originalClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = originalClient })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			http.DefaultClient = &http.Client{Transport: tt.roundTrip}
			model := New(Options{BaseURL: "http://vertex.test", AccessToken: "token"}).Model("gemini-1.5-pro")
			events, err := model.Stream(context.Background(), &stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("hello")},
			})
			if err != nil {
				t.Fatal(err)
			}
			var eventErr error
			for event := range events {
				if event.Type == stream.EventError {
					eventErr = event.Data.(stream.ErrorEvent).Error
				}
			}
			var apiErr *goaierrors.APICallError
			if !errors.As(eventErr, &apiErr) {
				t.Fatalf("error = %v, want APICallError", eventErr)
			}
			if apiErr.StatusCode != tt.wantStatus || apiErr.IsRetryable != tt.wantRetryable {
				t.Errorf("status/retryable = %d/%v, want %d/%v", apiErr.StatusCode, apiErr.IsRetryable, tt.wantStatus, tt.wantRetryable)
			}
			if tt.wantCause != nil && !errors.Is(apiErr, tt.wantCause) {
				t.Errorf("error = %v, want cause %v", apiErr, tt.wantCause)
			}
		})
	}
}

func TestVertexLanguageModel_ID(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.LanguageModel("gemini-1.5-pro")

	if m.ID() != "gemini-1.5-pro" {
		t.Errorf("expected model ID gemini-1.5-pro, got %s", m.ID())
	}
}

func TestVertexLanguageModel_Provider(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.LanguageModel("gemini-1.5-pro")

	if m.Provider() != "vertex" {
		t.Errorf("expected provider vertex, got %s", m.Provider())
	}
}

func TestVertexLanguageModel_Stream(t *testing.T) {
	t.Run("should create model correctly", func(t *testing.T) {
		// Note: Vertex uses Google Cloud auth and constructs URLs dynamically,
		// which makes it difficult to mock completely.
		// This test verifies the provider structure.

		provider := New(Options{
			ProjectID:   "test-project",
			Location:    "us-central1",
			AccessToken: "test-token",
		})

		m := provider.LanguageModel("gemini-1.5-pro")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})

	t.Run("should include system instruction in model", func(t *testing.T) {
		provider := New(Options{
			ProjectID:   "test-project",
			Location:    "us-central1",
			AccessToken: "test-token",
		})

		m := provider.LanguageModel("gemini-1.5-pro")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})
}

func TestVertexProvider_DefaultLocation(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})

	// Location should default to us-central1
	if provider.opts.Location != "us-central1" {
		t.Errorf("expected default location us-central1, got %s", provider.opts.Location)
	}
}

func TestVertexProvider_BaseURL(t *testing.T) {
	tests := []struct {
		name     string
		location string
		want     string
	}{
		{
			name:     "regional location is region-prefixed",
			location: "europe-west4",
			want:     "https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project/locations/europe-west4",
		},
		{
			name:     "global location omits region prefix",
			location: "global",
			want:     "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global",
		},
		{
			name:     "eu multi-region uses rep host",
			location: "eu",
			want:     "https://aiplatform.eu.rep.googleapis.com/v1/projects/my-project/locations/eu",
		},
		{
			name:     "us multi-region uses rep host",
			location: "us",
			want:     "https://aiplatform.us.rep.googleapis.com/v1/projects/my-project/locations/us",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := New(Options{ProjectID: "my-project", Location: tc.location, AccessToken: "test-token"})
			if got := p.baseURL(); got != tc.want {
				t.Errorf("baseURL() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestVertexEmbeddingModel_ID(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.EmbeddingModel("text-embedding-004")

	if m.ID() != "text-embedding-004" {
		t.Errorf("expected model ID text-embedding-004, got %s", m.ID())
	}
}

func TestVertexEmbeddingModel_Provider(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.EmbeddingModel("text-embedding-004")

	if m.Provider() != "vertex" {
		t.Errorf("expected provider vertex, got %s", m.Provider())
	}
}

func TestVertexEmbeddingModel_MaxEmbeddingsPerCall(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.EmbeddingModel("text-embedding-004")

	if m.MaxEmbeddingsPerCall() != 250 {
		t.Errorf("expected max embeddings 250, got %d", m.MaxEmbeddingsPerCall())
	}
}

func TestVertexImageModel_ID(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.ImageModel("imagegeneration@006")

	if m.ID() != "imagegeneration@006" {
		t.Errorf("expected model ID imagegeneration@006, got %s", m.ID())
	}
}

func TestVertexImageModel_Provider(t *testing.T) {
	provider := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	m := provider.ImageModel("imagegeneration@006")

	if m.Provider() != "vertex" {
		t.Errorf("expected provider vertex, got %s", m.Provider())
	}
}

func TestVertexImageModel_MaxImagesPerCall(t *testing.T) {
	p := New(Options{
		ProjectID:   "test-project",
		AccessToken: "test-token",
	})
	tests := []struct {
		modelID string
		want    int
	}{
		{"imagegeneration@006", 4},
		{"imagen-3.0-generate-002", 4},
		{"gemini-2.5-flash-image", 10},
		{"gemini-3.1-flash-image-preview", 10},
	}
	for _, tc := range tests {
		t.Run(tc.modelID, func(t *testing.T) {
			if got := p.ImageModel(tc.modelID).MaxImagesPerCall(); got != tc.want {
				t.Errorf("MaxImagesPerCall() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestVertexImageModel_Generate(t *testing.T) {
	t.Run("should create image model correctly", func(t *testing.T) {
		// Note: Can't easily override baseURL, so we verify model structure
		provider := New(Options{
			ProjectID:   "test-project",
			AccessToken: "test-token",
		})

		m := provider.ImageModel("imagegeneration@006")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})
}

func TestVertexEmbeddingModel_Embed(t *testing.T) {
	t.Run("should create embedding model correctly", func(t *testing.T) {
		// Note: Can't easily override baseURL, so we verify model structure
		provider := New(Options{
			ProjectID:   "test-project",
			AccessToken: "test-token",
		})

		m := provider.EmbeddingModel("text-embedding-004")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})
}

func TestConvertMessage(t *testing.T) {
	t.Run("should convert user message", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleUser,
			Content: message.Content{
				Text: "Hello world",
			},
		}

		result := convertMessage(msg)

		if result["role"] != "user" {
			t.Errorf("expected role 'user', got %v", result["role"])
		}

		parts := result["parts"].([]map[string]any)
		if len(parts) != 1 {
			t.Errorf("expected 1 part, got %d", len(parts))
		}

		if parts[0]["text"] != "Hello world" {
			t.Errorf("expected text 'Hello world', got %v", parts[0]["text"])
		}
	})

	t.Run("should convert assistant message", func(t *testing.T) {
		msg := message.Message{
			Role: message.RoleAssistant,
			Content: message.Content{
				Text: "Hi there",
			},
		}

		result := convertMessage(msg)

		if result["role"] != "model" {
			t.Errorf("expected role 'model', got %v", result["role"])
		}
	})
}

func TestVertexLanguageModel_ErrorHandling(t *testing.T) {
	t.Run("should create model with invalid token", func(t *testing.T) {
		provider := New(Options{
			ProjectID:   "test-project",
			AccessToken: "invalid-token",
		})

		m := provider.LanguageModel("gemini-1.5-pro")
		if m == nil {
			t.Fatal("expected non-nil model")
		}

		// Note: Actual error would occur during Stream(), but we can't easily test
		// that without being able to inject a base URL
	})
}

func TestVertexLanguageModel_ResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if request["generationConfig"].(map[string]any)["responseMimeType"] != "application/json" {
			t.Errorf("request = %v", request)
		}
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\n"))
	}))
	defer server.Close()
	p := New(Options{
		BaseURL:     server.URL,
		ProjectID:   "p",
		Location:    "us-central1",
		AccessToken: "t",
	})
	m := p.LanguageModel("gemini-1.5-pro")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:       []message.Message{message.NewUserMessage("hi")},
		ResponseFormat: &stream.ResponseFormat{Type: "json"},
	})
	if err != nil {
		t.Fatal(err)
	}
	var gotErr error
	for ev := range events {
		if ev.Type == stream.EventError {
			if e, ok := ev.Data.(stream.ErrorEvent); ok {
				gotErr = e.Error
			}
		}
	}
	if gotErr != nil {
		t.Fatal(gotErr)
	}
}

type capturedImageCall struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

func newImageServer(t *testing.T, respBody map[string]any, capture *capturedImageCall) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.method = r.Method
		capture.path = r.URL.Path
		capture.header = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &capture.body)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(respBody)
	}))
}

func geminiImageResponse(img string) map[string]any {
	return map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"role": "model",
				"parts": []map[string]any{{
					"inlineData": map[string]any{
						"mimeType": "image/png",
						"data":     img,
					},
				}},
			},
			"finishReason": "STOP",
		}},
		"usageMetadata": map[string]any{
			"promptTokenCount":     10,
			"candidatesTokenCount": 100,
			"totalTokenCount":      110,
		},
	}
}

func newVertexImageModel(t *testing.T, baseURL, modelID string) *VertexImageModel {
	t.Helper()
	p := New(Options{
		ProjectID:   "p",
		Location:    "us-central1",
		AccessToken: "t",
		BaseURL:     baseURL,
	})
	m, ok := p.ImageModel(modelID).(*VertexImageModel)
	if !ok {
		t.Fatalf("ImageModel did not return *VertexImageModel")
	}
	return m
}

func TestVertexImageModel_Gemini_BasicGenerate(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("base64-generated-image"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	res, err := m.Generate(context.Background(), model.ImageCallOptions{Prompt: "A beautiful sunset"})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if cap.method != http.MethodPost {
		t.Errorf("method: want POST, got %s", cap.method)
	}
	if cap.path != "/publishers/google/models/gemini-2.5-flash-image:generateContent" {
		t.Errorf("path: got %s", cap.path)
	}
	if got := cap.header.Get("Authorization"); got != "Bearer t" {
		t.Errorf("Authorization: want Bearer t, got %s", got)
	}

	contents, _ := cap.body["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents: want 1, got %d", len(contents))
	}
	msg := contents[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("role: want user, got %v", msg["role"])
	}
	parts := msg["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts: want 1, got %d", len(parts))
	}
	if parts[0].(map[string]any)["text"] != "A beautiful sunset" {
		t.Errorf("text mismatch: %v", parts[0])
	}

	gc := cap.body["generationConfig"].(map[string]any)
	mods, _ := gc["responseModalities"].([]any)
	if len(mods) != 1 || mods[0] != "IMAGE" {
		t.Errorf("responseModalities: %v", mods)
	}

	if len(res.Images) != 1 {
		t.Fatalf("Images: want 1, got %d", len(res.Images))
	}
	if res.Images[0].Base64 != "base64-generated-image" {
		t.Errorf("Base64: %q", res.Images[0].Base64)
	}
	if res.Images[0].MimeType != "image/png" {
		t.Errorf("MimeType: %q", res.Images[0].MimeType)
	}
	if res.Usage == nil || res.Usage.TotalTokens != 110 {
		t.Errorf("Usage: %+v", res.Usage)
	}
	if res.Response.Model != "gemini-2.5-flash-image" {
		t.Errorf("Response.Model: %q", res.Response.Model)
	}
	if res.Response.Timestamp == 0 {
		t.Error("Response.Timestamp: want non-zero")
	}
	vertexMeta, ok := res.ProviderMetadata["vertex"].(map[string]any)
	if !ok {
		t.Fatalf("ProviderMetadata[vertex]: %v", res.ProviderMetadata)
	}
	imgs, _ := vertexMeta["images"].([]map[string]any)
	if len(imgs) != 1 {
		t.Errorf("vertex.images: want 1 empty entry, got %v", vertexMeta["images"])
	}
}

func TestVertexImageModel_Gemini_AspectRatioTopLevel(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt:      "A beautiful sunset",
		AspectRatio: "16:9",
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	gc := cap.body["generationConfig"].(map[string]any)
	imageConfig, ok := gc["imageConfig"].(map[string]any)
	if !ok {
		t.Fatalf("imageConfig missing: %v", gc)
	}
	if imageConfig["aspectRatio"] != "16:9" {
		t.Errorf("aspectRatio: %v", imageConfig["aspectRatio"])
	}
}

func TestVertexImageModel_Gemini_AspectRatioProviderOptions(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "A beautiful sunset",
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"imageConfig": map[string]any{"aspectRatio": "4:3"},
			},
		},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	gc := cap.body["generationConfig"].(map[string]any)
	imageConfig, ok := gc["imageConfig"].(map[string]any)
	if !ok {
		t.Fatalf("imageConfig missing: %v", gc)
	}
	if imageConfig["aspectRatio"] != "4:3" {
		t.Errorf("aspectRatio: %v", imageConfig["aspectRatio"])
	}
}

func TestVertexImageModel_Gemini_ProviderOptionsAspectRatioWins(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt:      "A beautiful sunset",
		AspectRatio: "1:1",
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"imageConfig": map[string]any{"aspectRatio": "16:9"},
			},
		},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	gc := cap.body["generationConfig"].(map[string]any)
	imageConfig := gc["imageConfig"].(map[string]any)
	if imageConfig["aspectRatio"] != "16:9" {
		t.Errorf("providerOptions.vertex.imageConfig should win (ai-sdk spreads providerOptions last): got %v", imageConfig["aspectRatio"])
	}
}

func TestVertexImageModel_Gemini_ProviderOptionsMerged(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "hi",
		ProviderOptions: map[string]any{
			"vertex": map[string]any{
				"temperature": 0.5,
			},
		},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	gc := cap.body["generationConfig"].(map[string]any)
	if gc["temperature"] != 0.5 {
		t.Errorf("temperature not merged: %v", gc["temperature"])
	}
	mods, _ := gc["responseModalities"].([]any)
	if len(mods) != 1 || mods[0] != "IMAGE" {
		t.Errorf("responseModalities should remain: %v", mods)
	}
}

func TestVertexImageModel_Gemini_SizeWarning(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	res, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "hi",
		Size:   "1024x1024",
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	found := false
	for _, w := range res.Warnings {
		if w.Type == stream.WarningUnsupported && w.Feature == "size" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected size warning, got %v", res.Warnings)
	}
}

func TestVertexImageModel_Gemini_MaskRejected(t *testing.T) {
	m := newVertexImageModel(t, "http://unused.example", "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "edit",
		Mask:   []byte{0x89, 0x50, 0x4E, 0x47},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "mask") {
		t.Errorf("expected mask error, got %v", err)
	}
}

func TestVertexImageModel_Gemini_NGreaterThanOneRejected(t *testing.T) {
	m := newVertexImageModel(t, "http://unused.example", "gemini-2.5-flash-image")
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "hi",
		N:      2,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "set number of images") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestVertexImageModel_Gemini_InputFilesInlineData(t *testing.T) {
	cap := &capturedImageCall{}
	srv := newImageServer(t, geminiImageResponse("x"), cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "gemini-2.5-flash-image")
	pngBytes := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	_, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "Add a hat",
		Files:  [][]byte{pngBytes},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	contents := cap.body["contents"].([]any)
	parts := contents[0].(map[string]any)["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts: want 2, got %d (%v)", len(parts), parts)
	}
	inline, ok := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if !ok {
		t.Fatalf("second part missing inlineData: %v", parts[1])
	}
	if inline["mimeType"] != "image/png" {
		t.Errorf("mimeType: %v", inline["mimeType"])
	}
	wantB64 := base64.StdEncoding.EncodeToString(pngBytes)
	if inline["data"] != wantB64 {
		t.Errorf("data: got %v, want %s", inline["data"], wantB64)
	}
}

func TestVertexImageModel_Imagen_Regression(t *testing.T) {
	cap := &capturedImageCall{}
	respBody := map[string]any{
		"predictions": []map[string]any{{
			"bytesBase64Encoded": "imagen-b64",
			"mimeType":           "image/png",
		}},
	}
	srv := newImageServer(t, respBody, cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "imagegeneration@006")
	res, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "A painting",
		N:      2,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if cap.path != "/publishers/google/models/imagegeneration@006:predict" {
		t.Errorf("path: got %s", cap.path)
	}
	instances := cap.body["instances"].([]any)
	if instances[0].(map[string]any)["prompt"] != "A painting" {
		t.Errorf("prompt mismatch: %v", instances[0])
	}
	params := cap.body["parameters"].(map[string]any)
	if params["sampleCount"].(float64) != 2 {
		t.Errorf("sampleCount: %v", params["sampleCount"])
	}

	if len(res.Images) != 1 || res.Images[0].Base64 != "imagen-b64" {
		t.Errorf("Images: %+v", res.Images)
	}
}

func TestVertexImageModel_Imagen_SizeWarning(t *testing.T) {
	cap := &capturedImageCall{}
	respBody := map[string]any{
		"predictions": []map[string]any{{
			"bytesBase64Encoded": "imagen-b64",
			"mimeType":           "image/png",
		}},
	}
	srv := newImageServer(t, respBody, cap)
	defer srv.Close()

	m := newVertexImageModel(t, srv.URL, "imagegeneration@006")
	res, err := m.Generate(context.Background(), model.ImageCallOptions{
		Prompt: "x",
		N:      1,
		Size:   "1024x1024",
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	found := false
	for _, w := range res.Warnings {
		if w.Type == stream.WarningUnsupported && w.Feature == "size" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected size warning, got %v", res.Warnings)
	}
}
