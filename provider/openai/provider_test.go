package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/openai/src/chat/openai-chat-language-model.test.ts

func TestOpenAIProvider_ID(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})

	if p.ID() != "openai" {
		t.Errorf("expected provider ID openai, got %s", p.ID())
	}
}

func TestOpenAIProvider_Models(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})

	models := p.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasGPT := false
	for _, m := range models {
		if strings.Contains(m, "gpt") {
			hasGPT = true
		}
	}
	if !hasGPT {
		t.Error("expected 'gpt' model in models list")
	}
}

func TestOpenAIChatModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
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

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, err := m.Stream(context.Background(), &stream.CallOptions{
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
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":30,"total_tokens":34}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, err := m.Stream(context.Background(), &stream.CallOptions{
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

		if usage.InputTotal() != 4 {
			t.Errorf("expected prompt tokens 4, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 30 {
			t.Errorf("expected completion tokens 30, got %d", usage.OutputTotal())
		}
	})
}

func TestOpenAIChatModel_Headers(t *testing.T) {
	t.Run("should pass authorization header", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, _ := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Drain events
		for range events {
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got %s", receivedHeaders.Get("Authorization"))
		}
	})

	t.Run("should pass custom headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, _ := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		// Drain events
		for range events {
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestOpenAIChatModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, _ := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Drain events
		for range events {
		}

		if receivedBody["model"] != "gpt-3.5-turbo" {
			t.Errorf("expected model gpt-3.5-turbo, got %v", receivedBody["model"])
		}

		if receivedBody["stream"] != true {
			t.Error("expected stream to be true")
		}

		messages, ok := receivedBody["messages"].([]any)
		if !ok || len(messages) == 0 {
			t.Error("expected messages in request body")
		}
	})
}

func TestOpenAIChatModel_ToolCalls(t *testing.T) {
	t.Run("should extract tool calls", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1680003600,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"San Francisco\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("What's the weather in San Francisco?"),
			},
			Tools: []tool.Tool{
				{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string"}}}`),
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var toolCalls []stream.ToolCallEvent
		for event := range events {
			if event.Type == stream.EventToolCall {
				if tc, ok := event.Data.(stream.ToolCallEvent); ok {
					toolCalls = append(toolCalls, tc)
				}
			}
		}

		if len(toolCalls) != 1 {
			t.Errorf("expected 1 tool call, got %d", len(toolCalls))
		}

		if toolCalls[0].ToolName != "get_weather" {
			t.Errorf("expected tool name get_weather, got %s", toolCalls[0].ToolName)
		}
	})
}

func TestOpenAIChatModel_ErrorResponse(t *testing.T) {
	t.Run("should emit error event on API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`))
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "invalid-key",
			BaseURL: server.URL,
		})
		m := p.Chat("gpt-3.5-turbo")

		events, err := m.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			return // Error returned directly - test passes
		}

		var gotError bool
		for event := range events {
			if event.Type == stream.EventError {
				gotError = true
				break
			}
		}

		if !gotError {
			t.Error("expected error event in stream")
		}
	})
}

// Image model tests - translated from openai-image-model.test.ts
func TestOpenAIImageModel_ID(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.ImageModel("dall-e-3")

	if m.ID() != "dall-e-3" {
		t.Errorf("expected model ID dall-e-3, got %s", m.ID())
	}
}

func TestOpenAIImageModel_Provider(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.ImageModel("dall-e-3")

	if m.Provider() != "openai" {
		t.Errorf("expected provider openai, got %s", m.Provider())
	}
}

func TestOpenAIImageModel_MaxImagesPerCall(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})

	dalle3 := p.ImageModel("dall-e-3")
	if dalle3.MaxImagesPerCall() != 1 {
		t.Errorf("expected DALL-E 3 max images 1, got %d", dalle3.MaxImagesPerCall())
	}

	dalle2 := p.ImageModel("dall-e-2")
	if dalle2.MaxImagesPerCall() != 10 {
		t.Errorf("expected DALL-E 2 max images 10, got %d", dalle2.MaxImagesPerCall())
	}
}

func TestOpenAIImageModel_Generate(t *testing.T) {
	t.Run("should generate image with prompt", func(t *testing.T) {
		var receivedBody map[string]any
		testImageData := []byte("test-image-data")
		b64Image := base64.StdEncoding.EncodeToString(testImageData)

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1680003600,
				"data": []map[string]any{
					{"b64_json": b64Image},
				},
			})
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.ImageModel("dall-e-3")

		result, err := m.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "A beautiful sunset over the ocean",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["prompt"] != "A beautiful sunset over the ocean" {
			t.Errorf("expected prompt 'A beautiful sunset over the ocean', got %v", receivedBody["prompt"])
		}

		if len(result.Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(result.Images))
		}
	})

	t.Run("should pass size option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"created": 1680003600,
				"data": []map[string]any{
					{"b64_json": "dGVzdA=="},
				},
			})
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.ImageModel("dall-e-3")

		_, err := m.Generate(context.Background(), model.ImageCallOptions{
			Prompt: "test",
			Size:   "1792x1024",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["size"] != "1792x1024" {
			t.Errorf("expected size 1792x1024, got %v", receivedBody["size"])
		}
	})
}

// Embedding model tests - translated from openai-embedding-model.test.ts
func TestOpenAIEmbeddingModel_ID(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.EmbeddingModel("text-embedding-3-small")

	if m.ID() != "text-embedding-3-small" {
		t.Errorf("expected model ID text-embedding-3-small, got %s", m.ID())
	}
}

func TestOpenAIEmbeddingModel_Provider(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.EmbeddingModel("text-embedding-3-small")

	if m.Provider() != "openai" {
		t.Errorf("expected provider openai, got %s", m.Provider())
	}
}

func TestOpenAIEmbeddingModel_Embed(t *testing.T) {
	t.Run("should generate embeddings", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{
					{
						"embedding": []float64{0.1, 0.2, 0.3},
						"index":     0,
					},
				},
				"usage": map[string]any{
					"prompt_tokens": 5,
					"total_tokens":  5,
				},
			})
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.EmbeddingModel("text-embedding-3-small")

		result, err := m.Embed(context.Background(), model.EmbedCallOptions{
			Values: []string{"Hello, world!"},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 1 {
			t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
		}

		if len(result.Embeddings[0].Values) != 3 {
			t.Errorf("expected embedding length 3, got %d", len(result.Embeddings[0].Values))
		}

		if result.Usage.Tokens != 5 {
			t.Errorf("expected tokens 5, got %d", result.Usage.Tokens)
		}
	})
}

// Speech model tests - translated from openai-speech-model.test.ts
func TestOpenAISpeechModel_ID(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.SpeechModel("tts-1")

	if m.ID() != "tts-1" {
		t.Errorf("expected model ID tts-1, got %s", m.ID())
	}
}

func TestOpenAISpeechModel_Provider(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.SpeechModel("tts-1")

	if m.Provider() != "openai" {
		t.Errorf("expected provider openai, got %s", m.Provider())
	}
}

func TestOpenAISpeechModel_Generate(t *testing.T) {
	t.Run("should generate speech", func(t *testing.T) {
		var receivedBody map[string]any
		testAudioData := []byte("test-audio-data")

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "audio/mpeg")
			w.WriteHeader(http.StatusOK)
			w.Write(testAudioData)
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.SpeechModel("tts-1")

		result, err := m.Generate(context.Background(), model.SpeechCallOptions{
			Text:  "Hello, world!",
			Voice: "alloy",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["input"] != "Hello, world!" {
			t.Errorf("expected input 'Hello, world!', got %v", receivedBody["input"])
		}

		if receivedBody["voice"] != "alloy" {
			t.Errorf("expected voice 'alloy', got %v", receivedBody["voice"])
		}

		if len(result.Audio) != len(testAudioData) {
			t.Errorf("expected audio length %d, got %d", len(testAudioData), len(result.Audio))
		}
	})
}

// Transcription model tests - translated from openai-transcription-model.test.ts
func TestOpenAITranscriptionModel_ID(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.TranscriptionModel("whisper-1")

	if m.ID() != "whisper-1" {
		t.Errorf("expected model ID whisper-1, got %s", m.ID())
	}
}

func TestOpenAITranscriptionModel_Provider(t *testing.T) {
	p := New(provider.Options{APIKey: "test-key"})
	m := p.TranscriptionModel("whisper-1")

	if m.Provider() != "openai" {
		t.Errorf("expected provider openai, got %s", m.Provider())
	}
}

func TestOpenAITranscriptionModel_Transcribe(t *testing.T) {
	t.Run("should transcribe audio", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify multipart form
			if !strings.Contains(r.Header.Get("Content-Type"), "multipart/form-data") {
				t.Error("expected multipart/form-data content type")
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"text": "Hello, world!",
			})
		}))
		defer server.Close()

		p := New(provider.Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
		})
		m := p.TranscriptionModel("whisper-1")

		testAudioData := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		result, err := m.Transcribe(context.Background(), model.TranscribeCallOptions{
			Audio:    testAudioData,
			MimeType: "audio/wav",
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected text 'Hello, world!', got %s", result.Text)
		}
	})
}
