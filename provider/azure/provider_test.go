package azure

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/azure/src/azure-openai-provider.test.ts

func TestAzureProvider_ID(t *testing.T) {
	provider := New(Options{
		ResourceName: "test-resource",
		APIKey:       "test-key",
	})

	if provider.ID() != "azure" {
		t.Errorf("expected provider ID azure, got %s", provider.ID())
	}
}

func TestAzureProvider_Models(t *testing.T) {
	provider := New(Options{
		ResourceName: "test-resource",
		APIKey:       "test-key",
	})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}
}

func TestAzureModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify api-version query param
			if !strings.Contains(r.URL.RawQuery, "api-version=") {
				t.Error("expected api-version query parameter")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var textParts []string
		for event := range events {
			if event.Type == stream.EventTextDelta {
				if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
					textParts = append(textParts, delta.Text)
				}
			}
		}

		text := strings.Join(textParts, "")
		if text != "Hello, World!" {
			t.Errorf("expected text 'Hello, World!', got %s", text)
		}
	})

	t.Run("should extract usage", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var usage stream.Usage
		for event := range events {
			if event.Type == stream.EventFinish {
				if finish, ok := event.Data.(stream.FinishEvent); ok {
					usage = finish.Usage
				}
			}
		}

		if usage.InputTotal() != 10 {
			t.Errorf("expected prompt tokens 10, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 5 {
			t.Errorf("expected completion tokens 5, got %d", usage.OutputTotal())
		}
	})
}

func TestAzureModel_Headers(t *testing.T) {
	t.Run("should pass api-key header", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		model := provider.Model("gpt-4")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		// Azure uses api-key header instead of Authorization
		if receivedHeaders.Get("Api-Key") != "test-api-key" {
			t.Errorf("expected Api-Key header, got %s", receivedHeaders.Get("Api-Key"))
		}

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

// Embedding tests
var dummyEmbeddings = [][]float64{
	{0.1, 0.2, 0.3, 0.4, 0.5},
	{0.6, 0.7, 0.8, 0.9, 1.0},
}
var testEmbedValues = []string{"sunny day at the beach", "rainy day in the city"}

func TestAzureEmbedding_DoEmbed(t *testing.T) {
	t.Run("should extract embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify api-version query param
			if !strings.Contains(r.URL.RawQuery, "api-version=") {
				t.Error("expected api-version query parameter")
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
					{"object": "embedding", "index": 1, "embedding": dummyEmbeddings[1]},
				},
				"model": "text-embedding-ada-002",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		embModel := provider.EmbeddingModel("text-embedding-ada-002")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 2 {
			t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
		}

		for i, emb := range result.Embeddings {
			for j, v := range emb.Values {
				if v != dummyEmbeddings[i][j] {
					t.Errorf("expected embedding[%d][%d] = %f, got %f", i, j, dummyEmbeddings[i][j], v)
				}
			}
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": dummyEmbeddings[0]},
				},
				"model": "text-embedding-ada-002",
				"usage": map[string]int{"prompt_tokens": 8, "total_tokens": 8},
			})
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		embModel := provider.EmbeddingModel("text-embedding-ada-002")

		_, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues[:1],
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Api-Key") != "test-api-key" {
			t.Errorf("expected Api-Key header, got %s", receivedHeaders.Get("Api-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

// Image tests
func TestAzureImage_DoGenerate(t *testing.T) {
	t.Run("should generate image", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify api-version query param
			if !strings.Contains(r.URL.RawQuery, "api-version=") {
				t.Error("expected api-version query parameter")
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1733837122,
				"data": []map[string]any{
					{
						"revised_prompt": "A cute baby sea otter",
						"b64_json":       "base64-image-data",
					},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		imageModel := provider.ImageModel("dall-e-3")

		result, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "A cute baby sea otter",
			N:      1,
			Size:   "1024x1024",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}

		if result.Images[0].Base64 != "base64-image-data" {
			t.Errorf("expected base64 data, got %s", result.Images[0].Base64)
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1733837122,
				"data": []map[string]any{
					{"b64_json": "base64-image-data"},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		imageModel := provider.ImageModel("dall-e-3")

		_, err := imageModel.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Api-Key") != "test-api-key" {
			t.Errorf("expected Api-Key header, got %s", receivedHeaders.Get("Api-Key"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

// Transcription tests
func TestAzureTranscription_DoTranscribe(t *testing.T) {
	t.Run("should transcribe audio", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify api-version query param
			if !strings.Contains(r.URL.RawQuery, "api-version=") {
				t.Error("expected api-version query parameter")
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"text":     "Hello, world!",
				"language": "en",
				"duration": 2.5,
				"segments": []map[string]any{
					{"id": 0, "text": "Hello, world!", "start": 0.0, "end": 2.5},
				},
			})
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		transcriptionModel := provider.TranscriptionModel("whisper-1")

		result, err := transcriptionModel.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    []byte{1, 2, 3, 4, 5},
			Filename: "test.wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}

		if result.Language != "en" {
			t.Errorf("expected language en, got %s", result.Language)
		}
	})
}

// Speech tests
func TestAzureSpeech_DoGenerate(t *testing.T) {
	t.Run("should generate speech", func(t *testing.T) {
		testAudioData := []byte{1, 2, 3, 4, 5, 6, 7, 8}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify api-version query param
			if !strings.Contains(r.URL.RawQuery, "api-version=") {
				t.Error("expected api-version query parameter")
			}

			w.Header().Set("Content-Type", "audio/mpeg")
			w.Write(testAudioData)
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		speechModel := provider.SpeechModel("tts-1")

		result, err := speechModel.Generate(context.Background(), model.SpeechCallOptions{
			Text: "Hello, world!",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}

		if result.MimeType != "audio/mpeg" {
			t.Errorf("expected mime type audio/mpeg, got %s", result.MimeType)
		}
	})
}

func TestAzureModel_APIVersion(t *testing.T) {
	t.Run("should use default api version", func(t *testing.T) {
		var requestURL string

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestURL = r.URL.String()
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
		})
		for range events {
		}

		if !strings.Contains(requestURL, "api-version=") {
			t.Error("expected api-version in URL")
		}
	})

	t.Run("should use custom api version", func(t *testing.T) {
		var requestURL string

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestURL = r.URL.String()
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL:    server.URL,
			APIKey:     "test-api-key",
			APIVersion: "2025-04-01-preview",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
		})
		for range events {
		}

		if !strings.Contains(requestURL, "api-version=2025-04-01-preview") {
			t.Errorf("expected api-version=2025-04-01-preview in URL, got %s", requestURL)
		}
	})
}

// Tests for ProviderOptions - verifies ChatOptions are wired up correctly

func TestAzureModel_ProviderOptions(t *testing.T) {
	t.Run("should pass reasoning effort", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
			ProviderOptions: map[string]any{
				"reasoningEffort": "high",
			},
		})
		for range events {
		}

		if receivedBody["reasoning_effort"] != "high" {
			t.Errorf("expected reasoning_effort 'high', got %v", receivedBody["reasoning_effort"])
		}
	})

	t.Run("should pass reasoning summary", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
			ProviderOptions: map[string]any{
				"reasoningSummary": "detailed",
			},
		})
		for range events {
		}

		if receivedBody["reasoning_summary"] != "detailed" {
			t.Errorf("expected reasoning_summary 'detailed', got %v", receivedBody["reasoning_summary"])
		}
	})

	t.Run("should pass store option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
			ProviderOptions: map[string]any{
				"store": true,
			},
		})
		for range events {
		}

		if receivedBody["store"] != true {
			t.Errorf("expected store true, got %v", receivedBody["store"])
		}
	})

	t.Run("should pass user option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
			ProviderOptions: map[string]any{
				"user": "user_123",
			},
		})
		for range events {
		}

		if receivedBody["user"] != "user_123" {
			t.Errorf("expected user 'user_123', got %v", receivedBody["user"])
		}
	})

	t.Run("should pass parallel tool calls", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte(`data: {"id":"1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()

		provider := New(Options{
			BaseURL: server.URL,
			APIKey:  "test-api-key",
		})
		model := provider.Model("gpt-4")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("Hi")},
			ProviderOptions: map[string]any{
				"parallelToolCalls": false,
			},
		})
		for range events {
		}

		if receivedBody["parallel_tool_calls"] != false {
			t.Errorf("expected parallel_tool_calls false, got %v", receivedBody["parallel_tool_calls"])
		}
	})
}

func TestAzureModel_ResponseFormat(t *testing.T) {
	runCapture := func(t *testing.T, callOpts *stream.CallOptions) map[string]any {
		t.Helper()
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer server.Close()
		p := New(Options{
			APIKey:       "k",
			ResourceName: "r",
			BaseURL:      server.URL,
			APIVersion:   "2024-02-01",
		})
		m := p.Model("gpt-4o")
		if callOpts.Messages == nil {
			callOpts.Messages = []message.Message{message.NewUserMessage("hi")}
		}
		events, err := m.Stream(context.Background(), callOpts)
		if err != nil {
			t.Fatal(err)
		}
		for range events {
		}
		return body
	}

	t.Run("no schema maps to json_object", func(t *testing.T) {
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Errorf("expected json_object, got %v", rf)
		}
	})

	t.Run("schema maps to json_schema with strict from provider options", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object"}`)
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema, Name: "thing"},
			ProviderOptions: map[string]any{
				"strictJsonSchema": true,
			},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_schema" {
			t.Fatalf("expected json_schema, got %v", rf)
		}
		js, _ := rf["json_schema"].(map[string]any)
		if js["name"] != "thing" {
			t.Errorf("expected name=thing, got %v", js["name"])
		}
		if js["strict"] != true {
			t.Errorf("expected strict=true, got %v", js["strict"])
		}
	})
}
