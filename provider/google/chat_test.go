package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/google/src/google-generative-ai-language-model.test.ts

func createTestProvider(serverURL string) *Provider {
	return New(Options{
		APIKey:  "test-api-key",
		BaseURL: serverURL,
	})
}

func TestGoogleModel_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.Model("gemini-1.5-pro")

	if model.ID() != "gemini-1.5-pro" {
		t.Errorf("expected model ID gemini-1.5-pro, got %s", model.ID())
	}
}

func TestGoogleModel_Provider(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	model := provider.Model("gemini-1.5-pro")

	if model.Provider() != "google" {
		t.Errorf("expected provider google, got %s", model.Provider())
	}
}

func TestGoogleModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"finishReason":"","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0,"totalTokenCount":5}}`,
				`{"candidates":[{"content":{"parts":[{"text":", "}],"role":"model"},"finishReason":"","index":0}]}`,
				`{"candidates":[{"content":{"parts":[{"text":"World!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

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
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

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

func TestGoogleModel_RequestBody(t *testing.T) {
	t.Run("should send the model id and settings", func(t *testing.T) {
		var receivedBody map[string]any
		var requestURL string

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestURL = r.URL.Path
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		temp := 0.5
		topP := 0.9
		topK := 10
		maxTokens := 1000

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: &maxTokens,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		// Check URL contains the model
		if !strings.Contains(requestURL, "gemini-1.5-pro") {
			t.Errorf("expected URL to contain model, got %s", requestURL)
		}

		// Check generation config
		genConfig, ok := receivedBody["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("expected generationConfig in request")
		}

		if genConfig["temperature"] != 0.5 {
			t.Errorf("expected temperature 0.5, got %v", genConfig["temperature"])
		}
		if genConfig["topP"] != 0.9 {
			t.Errorf("expected topP 0.9, got %v", genConfig["topP"])
		}
		if genConfig["topK"] != float64(10) {
			t.Errorf("expected topK 10, got %v", genConfig["topK"])
		}
		if genConfig["maxOutputTokens"] != float64(1000) {
			t.Errorf("expected maxOutputTokens 1000, got %v", genConfig["maxOutputTokens"])
		}
	})
}

func TestGoogleModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		model := provider.Model("gemini-1.5-pro")

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

		// Check custom provider header
		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		// Check custom request header
		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestGoogleModel_ErrorResponse(t *testing.T) {
	t.Run("should handle API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"code":400,"message":"API key not valid","status":"INVALID_ARGUMENT"}}`))
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error creating stream: %v", err)
		}

		var errorEvent *stream.ErrorEvent
		for event := range events {
			if event.Type == stream.EventError {
				if e, ok := event.Data.(stream.ErrorEvent); ok {
					errorEvent = &e
				}
			}
		}

		if errorEvent == nil {
			t.Fatal("expected error event")
		}

		if errorEvent.Error == nil {
			t.Fatal("expected error in error event")
		}
	})
}

func TestGoogleModel_ToolCalls(t *testing.T) {
	t.Run("should extract tool calls", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"San Francisco"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
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

		var toolCallEvent *stream.ToolCallEvent
		for event := range events {
			if event.Type == stream.EventToolCall {
				if tc, ok := event.Data.(stream.ToolCallEvent); ok {
					toolCallEvent = &tc
				}
			}
		}

		if toolCallEvent == nil {
			t.Fatal("expected tool call event")
		}

		if toolCallEvent.ToolName != "get_weather" {
			t.Errorf("expected tool name get_weather, got %s", toolCallEvent.ToolName)
		}

		var input map[string]any
		if err := json.Unmarshal(toolCallEvent.Input, &input); err != nil {
			t.Fatalf("failed to unmarshal input: %v", err)
		}

		if input["location"] != "San Francisco" {
			t.Errorf("expected location San Francisco, got %v", input["location"])
		}
	})
}

func TestGoogleModel_FinishReason(t *testing.T) {
	tests := []struct {
		name           string
		finishReason   string
		expectedReason stream.FinishReason
	}{
		{"STOP should map to stop", "STOP", stream.FinishReasonStop},
		{"MAX_TOKENS should map to length", "MAX_TOKENS", stream.FinishReasonLength},
		{"SAFETY should map to content_filter", "SAFETY", stream.FinishReasonContentFilter},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)

				chunk := `{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"` + tc.finishReason + `","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`
				w.Write([]byte("data: " + chunk + "\n\n"))
			}))
			defer server.Close()

			provider := createTestProvider(server.URL)
			model := provider.Model("gemini-1.5-pro")

			events, err := model.Stream(context.Background(), &stream.CallOptions{
				Messages: []message.Message{
					message.NewUserMessage("Hello"),
				},
			})

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var finishReason stream.FinishReason
			for event := range events {
				if event.Type == stream.EventFinish {
					if f, ok := event.Data.(stream.FinishEvent); ok {
						finishReason = f.FinishReason
					}
				}
			}

			if finishReason != tc.expectedReason {
				t.Errorf("expected finish reason %v, got %v", tc.expectedReason, finishReason)
			}
		})
	}
}

func TestGoogleModel_SystemMessage(t *testing.T) {
	t.Run("should pass system instruction", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"totalTokenCount":11}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("You are a helpful assistant."),
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		// Check system instruction
		sysInstruction, ok := receivedBody["systemInstruction"].(map[string]any)
		if !ok {
			t.Fatal("expected systemInstruction in request")
		}

		parts, ok := sysInstruction["parts"].([]any)
		if !ok || len(parts) == 0 {
			t.Fatal("expected parts in systemInstruction")
		}

		firstPart, ok := parts[0].(map[string]any)
		if !ok {
			t.Fatal("expected first part to be an object")
		}

		if firstPart["text"] != "You are a helpful assistant." {
			t.Errorf("expected system message text, got %v", firstPart["text"])
		}
	})
}

// Tests for ProviderOptions

func TestGoogleModel_SafetySettings(t *testing.T) {
	t.Run("should send safetySettings provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"safetySettings": []map[string]any{
					{
						"category":  "HARM_CATEGORY_HATE_SPEECH",
						"threshold": "BLOCK_LOW_AND_ABOVE",
					},
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		safetySettings, ok := receivedBody["safetySettings"].([]any)
		if !ok {
			t.Fatal("expected safetySettings array in request")
		}
		if len(safetySettings) != 1 {
			t.Fatalf("expected 1 safety setting, got %d", len(safetySettings))
		}
		setting := safetySettings[0].(map[string]any)
		if setting["category"] != "HARM_CATEGORY_HATE_SPEECH" {
			t.Errorf("expected category 'HARM_CATEGORY_HATE_SPEECH', got %v", setting["category"])
		}
		if setting["threshold"] != "BLOCK_LOW_AND_ABOVE" {
			t.Errorf("expected threshold 'BLOCK_LOW_AND_ABOVE', got %v", setting["threshold"])
		}
	})
}

func TestGoogleModel_CachedContent(t *testing.T) {
	t.Run("should send cachedContent provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"cachedContent": "cachedContents/abc123",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if receivedBody["cachedContent"] != "cachedContents/abc123" {
			t.Errorf("expected cachedContent 'cachedContents/abc123', got %v", receivedBody["cachedContent"])
		}
	})
}

func TestGoogleModel_ThinkingConfig(t *testing.T) {
	t.Run("should send thinkingConfig provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-2.0-flash-thinking")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"thinkingConfig": map[string]any{
					"thinkingBudget": 2048,
					"thinkingLevel":  "high",
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		genConfig, ok := receivedBody["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("expected generationConfig in request")
		}
		thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]any)
		if !ok {
			t.Fatal("expected thinkingConfig in generationConfig")
		}
		if thinkingConfig["thinkingBudget"] != float64(2048) {
			t.Errorf("expected thinkingBudget 2048, got %v", thinkingConfig["thinkingBudget"])
		}
		if thinkingConfig["thinkingLevel"] != "high" {
			t.Errorf("expected thinkingLevel 'high', got %v", thinkingConfig["thinkingLevel"])
		}
	})
}

func TestGoogleModel_ResponseModalities(t *testing.T) {
	t.Run("should send responseModalities provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-2.0-flash")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"responseModalities": []string{"TEXT", "IMAGE"},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		genConfig, ok := receivedBody["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("expected generationConfig in request")
		}
		modalities, ok := genConfig["responseModalities"].([]any)
		if !ok {
			t.Fatal("expected responseModalities in generationConfig")
		}
		if len(modalities) != 2 {
			t.Errorf("expected 2 modalities, got %d", len(modalities))
		}
	})
}

func TestGoogleModel_MediaResolution(t *testing.T) {
	t.Run("should send mediaResolution provider option", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte("data: " + chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"mediaResolution": "MEDIA_RESOLUTION_HIGH",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		genConfig, ok := receivedBody["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("expected generationConfig in request")
		}
		if genConfig["mediaResolution"] != "MEDIA_RESOLUTION_HIGH" {
			t.Errorf("expected mediaResolution 'MEDIA_RESOLUTION_HIGH', got %v", genConfig["mediaResolution"])
		}
	})
}

func TestGoogleModel_ResponseFormat(t *testing.T) {
	runCapture := func(t *testing.T, callOpts *stream.CallOptions) map[string]any {
		t.Helper()
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}` + "\n\n"))
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("gemini-1.5-pro")
		if callOpts.Messages == nil {
			callOpts.Messages = []message.Message{message.NewUserMessage("hi")}
		}
		events, err := model.Stream(context.Background(), callOpts)
		if err != nil {
			t.Fatal(err)
		}
		for range events {
		}
		return body
	}

	t.Run("no schema sets only responseMimeType", func(t *testing.T) {
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		gc, _ := body["generationConfig"].(map[string]any)
		if gc["responseMimeType"] != "application/json" {
			t.Errorf("expected responseMimeType application/json, got %v", gc["responseMimeType"])
		}
		if _, has := gc["responseSchema"]; has {
			t.Errorf("expected no responseSchema when schema is nil, got %v", gc["responseSchema"])
		}
	})

	t.Run("schema is converted and set as responseSchema", func(t *testing.T) {
		schema := json.RawMessage(`{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`)
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})
		gc, _ := body["generationConfig"].(map[string]any)
		if gc["responseMimeType"] != "application/json" {
			t.Errorf("expected responseMimeType application/json, got %v", gc["responseMimeType"])
		}
		resp, ok := gc["responseSchema"].(map[string]any)
		if !ok {
			t.Fatalf("expected responseSchema object, got %v", gc["responseSchema"])
		}
		if _, has := resp["$schema"]; has {
			t.Errorf("expected $schema stripped, got %v", resp)
		}
		if resp["type"] != "object" {
			t.Errorf("expected type=object, got %v", resp)
		}
	})

	t.Run("StructuredOutputs=false skips responseSchema", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
		falseVal := false
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
			ProviderOptions: map[string]any{
				"structuredOutputs": falseVal,
			},
		})
		gc, _ := body["generationConfig"].(map[string]any)
		if gc["responseMimeType"] != "application/json" {
			t.Errorf("expected responseMimeType, got %v", gc["responseMimeType"])
		}
		if _, has := gc["responseSchema"]; has {
			t.Errorf("expected no responseSchema when StructuredOutputs=false, got %v", gc["responseSchema"])
		}
	})
}

// Regression for ai-sdk PRs #4e22c2c (serviceTier), #46a3584
// (streamFunctionCallArguments), and #2565e70 (searchTypes + timeRangeFilter
// on googleSearch).
func TestGoogleModel_NewOptionsOnWire(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}` + "\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.Model("gemini-3.1-pro-preview")

	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		Tools: []tool.Tool{GoogleSearchWith(GoogleSearchOptions{
			SearchTypes: &GoogleSearchTypes{ImageSearch: &struct{}{}},
			TimeRangeFilter: &GoogleSearchTimeRange{
				StartTime: "2026-01-01T00:00:00Z",
				EndTime:   "2026-02-01T00:00:00Z",
			},
		})},
		ProviderOptions: map[string]any{
			"serviceTier":                 "priority",
			"streamFunctionCallArguments": true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}

	gc, _ := captured["generationConfig"].(map[string]any)
	if gc == nil {
		t.Fatal("expected generationConfig on request")
	}
	if gc["serviceTier"] != "priority" {
		t.Errorf("serviceTier = %v, want priority", gc["serviceTier"])
	}
	if gc["streamFunctionCallArguments"] != true {
		t.Errorf("streamFunctionCallArguments = %v, want true", gc["streamFunctionCallArguments"])
	}

	tools, _ := captured["tools"].([]any)
	if len(tools) == 0 {
		t.Fatal("expected tools in request body")
	}
	t0, _ := tools[0].(map[string]any)
	gs, _ := t0["googleSearch"].(map[string]any)
	if gs == nil {
		t.Fatalf("expected googleSearch tool entry, got %v", t0)
	}
	st, _ := gs["searchTypes"].(map[string]any)
	if _, ok := st["imageSearch"]; !ok {
		t.Errorf("expected searchTypes.imageSearch, got %v", st)
	}
	tr, _ := gs["timeRangeFilter"].(map[string]any)
	if tr["startTime"] != "2026-01-01T00:00:00Z" {
		t.Errorf("timeRangeFilter.startTime = %v", tr["startTime"])
	}
}

// includeServerSideToolInvocations should ride along with toolConfig on
// non-Vertex Gemini (mirrors ai-sdk PR #14767). The Vertex endpoint rejects
// the field, but goai's vertex package is implemented separately and does
// not use this code path.
func TestGoogleModel_ToolConfigIncludesServerSideToolInvocations(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}` + "\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.Model("gemini-2.5-pro")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages:   []message.Message{message.NewUserMessage("hi")},
		ToolChoice: "required",
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}

	tc, ok := captured["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected toolConfig on request, got %T (%v)", captured["toolConfig"], captured["toolConfig"])
	}
	if tc["includeServerSideToolInvocations"] != true {
		t.Errorf("toolConfig.includeServerSideToolInvocations = %v, want true", tc["includeServerSideToolInvocations"])
	}
	fc, _ := tc["functionCallingConfig"].(map[string]any)
	if fc["mode"] != "ANY" {
		t.Errorf("functionCallingConfig.mode = %v, want ANY", fc["mode"])
	}
}

// Without ToolChoice no toolConfig should ride along — and therefore no
// includeServerSideToolInvocations either.
func TestGoogleModel_ToolConfigOmittedWhenNoToolChoice(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(`data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}` + "\n\n"))
	}))
	defer server.Close()

	p := New(Options{APIKey: "k", BaseURL: server.URL})
	m := p.Model("gemini-2.5-pro")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	if _, has := captured["toolConfig"]; has {
		t.Errorf("expected no toolConfig when ToolChoice is unset, got %v", captured["toolConfig"])
	}
}

// Verify the latest gemini 3.x + 2.5 model IDs landed and obsolete
// 1.0-pro is pruned.
func TestGoogleProvider_ModelsContainsLatest(t *testing.T) {
	p := New(Options{APIKey: "k"})
	have := map[string]bool{}
	for _, m := range p.Models() {
		have[m] = true
	}
	for _, w := range []string{
		"gemini-3.1-pro-preview",
		"gemini-3.1-flash-image-preview",
		"gemini-3.1-flash-lite-preview",
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-image",
	} {
		if !have[w] {
			t.Errorf("Models() missing %q", w)
		}
	}
	if have["gemini-1.0-pro"] {
		t.Error("Models() still lists retired gemini-1.0-pro")
	}
}
