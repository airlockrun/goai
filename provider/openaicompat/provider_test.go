package openaicompat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/openai-compatible/src/openai-compatible-provider.test.ts

func TestOpenAICompatProvider_ID(t *testing.T) {
	provider := New(Options{
		ProviderID: "custom-provider",
		APIKey:     "test-key",
		BaseURL:    "https://api.example.com/v1",
	})

	if provider.ID() != "custom-provider" {
		t.Errorf("expected provider ID custom-provider, got %s", provider.ID())
	}
}

func TestOpenAICompatModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"content":"World!"},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
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
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

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
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

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
		if usage.OutputTotal() != 20 {
			t.Errorf("expected completion tokens 20, got %d", usage.OutputTotal())
		}
	})
}

func TestOpenAICompatModel_Headers(t *testing.T) {
	t.Run("should pass default Bearer authorization header", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
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

	t.Run("should pass custom auth header", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
			AuthHeader: "X-Custom-Auth",
			AuthPrefix: "Key ",
		})
		model := provider.Model("custom-model")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Drain events
		for range events {
		}

		if receivedHeaders.Get("X-Custom-Auth") != "Key test-api-key" {
			t.Errorf("expected X-Custom-Auth header 'Key test-api-key', got %s", receivedHeaders.Get("X-Custom-Auth"))
		}
	})

	t.Run("should pass custom headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		model := provider.Model("custom-model")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
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

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestOpenAICompatModel_RequestBody(t *testing.T) {
	t.Run("should send the model and messages", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewDecoder(r.Body).Decode(&receivedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Drain events
		for range events {
		}

		if receivedBody["model"] != "custom-model" {
			t.Errorf("expected model custom-model, got %v", receivedBody["model"])
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

func TestOpenAICompatModel_ToolCalls(t *testing.T) {
	t.Run("should extract tool calls", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
				`data: {"id":"gen-id","object":"chat.completion.chunk","created":1680003600,"model":"custom-model","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"San Francisco\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "test-api-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("What's the weather in San Francisco?"),
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

func TestOpenAICompatModel_ResponseFormat(t *testing.T) {
	captureBody := func(t *testing.T) (*httptest.Server, *map[string]any) {
		t.Helper()
		body := map[string]any{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n"))
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		return server, &body
	}

	run := func(t *testing.T, opts Options, callOpts *stream.CallOptions) map[string]any {
		t.Helper()
		server, body := captureBody(t)
		defer server.Close()
		opts.ProviderID = "custom"
		opts.APIKey = "k"
		opts.BaseURL = server.URL
		model := New(opts).Model("custom-model")
		if callOpts.Messages == nil {
			callOpts.Messages = []message.Message{message.NewUserMessage("hi")}
		}
		events, err := model.Stream(context.Background(), callOpts)
		if err != nil {
			t.Fatal(err)
		}
		for range events {
		}
		return *body
	}

	t.Run("omits response_format when ResponseFormat nil", func(t *testing.T) {
		body := run(t, Options{}, &stream.CallOptions{})
		if _, has := body["response_format"]; has {
			t.Errorf("expected no response_format in request, got %v", body["response_format"])
		}
	})

	t.Run("no-schema maps to json_object", func(t *testing.T) {
		body := run(t, Options{}, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Errorf("expected json_object, got %v", rf)
		}
	})

	t.Run("schema without SupportsStructuredOutputs falls back to json_object and injects instruction", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
		body := run(t, Options{SupportsStructuredOutputs: false}, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Fatalf("expected json_object, got %v", rf)
		}
		messages, _ := body["messages"].([]any)
		if len(messages) == 0 {
			t.Fatal("expected messages")
		}
		first, _ := messages[0].(map[string]any)
		if first["role"] != "system" {
			t.Errorf("expected first message to be injected system, got %v", first)
		}
		content, _ := first["content"].(string)
		if !strings.Contains(content, "JSON schema:") || !strings.Contains(content, `"name"`) {
			t.Errorf("expected injected schema instruction, got %q", content)
		}
	})

	t.Run("schema with SupportsStructuredOutputs uses json_schema", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object"}`)
		body := run(t, Options{SupportsStructuredOutputs: true}, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema, Name: "thing"},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_schema" {
			t.Fatalf("expected json_schema, got %v", rf)
		}
		js, _ := rf["json_schema"].(map[string]any)
		if js["name"] != "thing" {
			t.Errorf("expected json_schema name 'thing', got %v", js["name"])
		}
		if js["schema"] == nil {
			t.Error("expected schema to be populated")
		}
	})

	t.Run("strictJsonSchema provider option flows into json_schema", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object"}`)
		body := run(t, Options{SupportsStructuredOutputs: true}, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
			ProviderOptions: map[string]any{
				"strictJsonSchema": true,
			},
		})
		rf, _ := body["response_format"].(map[string]any)
		js, _ := rf["json_schema"].(map[string]any)
		if js["strict"] != true {
			t.Errorf("expected strict=true, got %v", js["strict"])
		}
	})
}

func TestOpenAICompatModel_ErrorResponse(t *testing.T) {
	t.Run("should emit error event on API errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`))
		}))
		defer server.Close()

		provider := New(Options{
			ProviderID: "custom",
			APIKey:     "invalid-key",
			BaseURL:    server.URL,
		})
		model := provider.Model("custom-model")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
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

// Translated from ai-sdk PR #13006: fix(openai-compat): decode base64 string data
func TestConvertUserContent_TextFilePartBase64(t *testing.T) {
	content := message.Content{
		Parts: []message.Part{
			message.TextPart{Text: "Summarize this document"},
			message.FilePart{
				Data:     "UGxhaW4gdGV4dCBjb250ZW50", // base64("Plain text content")
				MimeType: "text/plain",
			},
		},
	}

	got := convertUserContent(content)
	parts, ok := got.([]chatContentPart)
	if !ok {
		t.Fatalf("expected []chatContentPart, got %T", got)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[1].Type != "text" {
		t.Errorf("expected text type for decoded file part, got %q", parts[1].Type)
	}
	if parts[1].Text != "Plain text content" {
		t.Errorf("expected decoded text %q, got %q", "Plain text content", parts[1].Text)
	}
}

// Translated from ai-sdk PR #12250: use looseObject for openaiCompatibleTokenUsageSchema
func TestOpenAICompatModel_PreservesExtraUsageFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := []string{
			`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
			`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":439,"total_tokens":457,"queue_time":0.061,"prompt_time":0.0002,"completion_time":0.798,"total_time":0.798}}`,
			`data: [DONE]`,
		}
		for _, c := range chunks {
			w.Write([]byte(c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	p := New(Options{ProviderID: "groq-like", APIKey: "k", BaseURL: server.URL})
	model := p.Model("m")
	events, err := model.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{{Role: message.RoleUser, Content: message.Content{Text: "hi"}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var finishMeta map[string]any
	for ev := range events {
		if ev.Type == stream.EventFinish {
			finishMeta = ev.Data.(stream.FinishEvent).ProviderMetadata
		}
	}

	if finishMeta == nil {
		t.Fatal("expected providerMetadata on finish event")
	}
	compat, ok := finishMeta["openaiCompat"].(map[string]any)
	if !ok {
		t.Fatalf("expected openaiCompat key, got %T", finishMeta["openaiCompat"])
	}
	raw, ok := compat["usageRaw"].(map[string]any)
	if !ok {
		t.Fatalf("expected usageRaw map, got %T", compat["usageRaw"])
	}
	if _, ok := raw["queue_time"]; !ok {
		t.Errorf("expected queue_time in usageRaw, got keys: %v", keys(raw))
	}
	if _, ok := raw["completion_time"]; !ok {
		t.Errorf("expected completion_time in usageRaw, got keys: %v", keys(raw))
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
