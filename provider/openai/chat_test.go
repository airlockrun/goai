package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Test fixtures

func createTestProvider(baseURL string) *Provider {
	return New(provider.Options{
		APIKey:  "test-api-key",
		BaseURL: baseURL,
	})
}

func createStreamChunks(chunks []string, finishReason string) string {
	var result strings.Builder
	for i, chunk := range chunks {
		data := map[string]any{
			"id":      "chatcmpl-test123",
			"object":  "chat.completion.chunk",
			"created": 1711115037,
			"model":   "gpt-3.5-turbo",
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]any{
						"content": chunk,
					},
				},
			},
		}
		// Add finish_reason to last chunk
		if i == len(chunks)-1 {
			data["choices"].([]map[string]any)[0]["finish_reason"] = finishReason
		}
		jsonData, _ := json.Marshal(data)
		result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))
	}
	// Add usage chunk
	usageData := map[string]any{
		"id":      "chatcmpl-test123",
		"object":  "chat.completion.chunk",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo",
		"choices": []map[string]any{},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 20,
			"total_tokens":      30,
		},
	}
	jsonData, _ := json.Marshal(usageData)
	result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))
	result.WriteString("data: [DONE]\n\n")
	return result.String()
}

func createToolCallStreamChunks() string {
	var result strings.Builder

	// First chunk: tool call start
	chunk1 := map[string]any{
		"id":      "chatcmpl-test123",
		"object":  "chat.completion.chunk",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo",
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []map[string]any{
						{
							"index": 0,
							"id":    "call_abc123",
							"type":  "function",
							"function": map[string]any{
								"name":      "get_weather",
								"arguments": "",
							},
						},
					},
				},
			},
		},
	}
	jsonData, _ := json.Marshal(chunk1)
	result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))

	// Second chunk: arguments part 1
	chunk2 := map[string]any{
		"id":      "chatcmpl-test123",
		"object":  "chat.completion.chunk",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo",
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []map[string]any{
						{
							"index": 0,
							"function": map[string]any{
								"arguments": `{"location":`,
							},
						},
					},
				},
			},
		},
	}
	jsonData, _ = json.Marshal(chunk2)
	result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))

	// Third chunk: arguments part 2
	chunk3 := map[string]any{
		"id":      "chatcmpl-test123",
		"object":  "chat.completion.chunk",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo",
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []map[string]any{
						{
							"index": 0,
							"function": map[string]any{
								"arguments": `"San Francisco"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
	}
	jsonData, _ = json.Marshal(chunk3)
	result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))

	// Usage chunk
	usageData := map[string]any{
		"id":      "chatcmpl-test123",
		"object":  "chat.completion.chunk",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo",
		"choices": []map[string]any{},
		"usage": map[string]any{
			"prompt_tokens":     15,
			"completion_tokens": 25,
			"total_tokens":      40,
		},
	}
	jsonData, _ = json.Marshal(usageData)
	result.WriteString(fmt.Sprintf("data: %s\n\n", jsonData))
	result.WriteString("data: [DONE]\n\n")

	return result.String()
}

// Tests

func TestChatModel_ID(t *testing.T) {
	p := createTestProvider("http://localhost")
	model := p.Chat("gpt-4o")

	if model.ID() != "gpt-4o" {
		t.Errorf("expected model ID 'gpt-4o', got '%s'", model.ID())
	}
}

func TestChatModel_Provider(t *testing.T) {
	p := createTestProvider("http://localhost")
	model := p.Chat("gpt-4o").(*ChatModel)

	if model.Provider() != "openai.chat" {
		t.Errorf("expected provider 'openai.chat', got '%s'", model.Provider())
	}
}

func TestChatModel_StreamText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path '/chat/completions', got '%s'", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got '%s'", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type 'application/json', got '%s'", r.Header.Get("Content-Type"))
		}

		// Verify request body
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		json.Unmarshal(body, &reqBody)

		if reqBody["model"] != "gpt-3.5-turbo" {
			t.Errorf("expected model 'gpt-3.5-turbo', got '%v'", reqBody["model"])
		}
		if reqBody["stream"] != true {
			t.Errorf("expected stream true, got '%v'", reqBody["stream"])
		}

		// Send streaming response
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := createStreamChunks([]string{"Hello", ", ", "World", "!"}, "stop")
		w.Write([]byte(chunks))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, err := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Say hello"),
		},
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var textDeltas []string
	var usage stream.Usage
	var finishReason stream.FinishReason

	for event := range events {
		switch e := event.Data.(type) {
		case stream.TextDeltaEvent:
			textDeltas = append(textDeltas, e.Text)
		case stream.FinishEvent:
			usage = e.Usage
			finishReason = e.FinishReason
		case stream.ErrorEvent:
			t.Fatalf("Unexpected error: %v", e.Error)
		}
	}

	// Verify text content
	fullText := strings.Join(textDeltas, "")
	if fullText != "Hello, World!" {
		t.Errorf("expected text 'Hello, World!', got '%s'", fullText)
	}

	// Verify usage
	if usage.InputTotal() != 10 {
		t.Errorf("expected prompt_tokens 10, got %d", usage.InputTotal())
	}
	if usage.OutputTotal() != 20 {
		t.Errorf("expected completion_tokens 20, got %d", usage.OutputTotal())
	}
	if usage.GrandTotal() != 30 {
		t.Errorf("expected total_tokens 30, got %d", usage.GrandTotal())
	}

	// Verify finish reason
	if finishReason != stream.FinishReasonStop {
		t.Errorf("expected finish_reason 'stop', got '%s'", finishReason)
	}
}

func TestChatModel_StreamWithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createToolCallStreamChunks()))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	// Create a tool (without execute function to avoid actual execution)
	tools := []tool.Tool{
		{
			Name:        "get_weather",
			Description: "Get weather for a location",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		},
	}

	ctx := context.Background()
	events, err := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("What's the weather in San Francisco?"),
		},
		Tools: tools,
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var toolInputStarted bool
	var toolInputDeltas []string
	var toolInputEnded bool
	var toolCallEvent *stream.ToolCallEvent
	var finishReason stream.FinishReason

	for event := range events {
		switch e := event.Data.(type) {
		case stream.ToolInputStartEvent:
			toolInputStarted = true
			if e.ToolName != "get_weather" {
				t.Errorf("expected tool name 'get_weather', got '%s'", e.ToolName)
			}
		case stream.ToolInputDeltaEvent:
			toolInputDeltas = append(toolInputDeltas, e.Delta)
		case stream.ToolInputEndEvent:
			toolInputEnded = true
		case stream.ToolCallEvent:
			toolCallEvent = &e
		case stream.FinishEvent:
			finishReason = e.FinishReason
		case stream.ErrorEvent:
			t.Fatalf("Unexpected error: %v", e.Error)
		}
	}

	if !toolInputStarted {
		t.Error("expected tool input start event")
	}
	if !toolInputEnded {
		t.Error("expected tool input end event")
	}
	if toolCallEvent == nil {
		t.Fatal("expected tool call event")
	}
	if toolCallEvent.ToolName != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got '%s'", toolCallEvent.ToolName)
	}
	if toolCallEvent.ToolCallID != "call_abc123" {
		t.Errorf("expected tool call ID 'call_abc123', got '%s'", toolCallEvent.ToolCallID)
	}

	// Verify accumulated arguments
	fullArgs := strings.Join(toolInputDeltas, "")
	if fullArgs != `{"location":"San Francisco"}` {
		t.Errorf("expected arguments '{\"location\":\"San Francisco\"}', got '%s'", fullArgs)
	}

	if finishReason != stream.FinishReasonToolCalls {
		t.Errorf("expected finish_reason 'tool_calls', got '%s'", finishReason)
	}
}

func TestChatModel_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"Invalid request","type":"invalid_request_error","code":"invalid_api_key"}}`))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, err := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var errorEvent *stream.ErrorEvent
	for event := range events {
		if e, ok := event.Data.(stream.ErrorEvent); ok {
			errorEvent = &e
		}
	}

	if errorEvent == nil {
		t.Fatal("expected error event")
	}
	if !strings.Contains(errorEvent.Error.Error(), "status 400") {
		t.Errorf("expected error to contain 'status 400', got '%s'", errorEvent.Error.Error())
	}
}

func TestChatModel_RequestWithTemperature(t *testing.T) {
	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{"Hi"}, "stop")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	temp := 0.7
	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
		Temperature: &temp,
	})

	// Consume events
	for range events {
	}

	if capturedBody["temperature"] != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", capturedBody["temperature"])
	}
}

func TestChatModel_RequestWithMaxTokens(t *testing.T) {
	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{"Hi"}, "stop")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	maxTokens := 100
	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
		MaxOutputTokens: &maxTokens,
	})

	// Consume events
	for range events {
	}

	if capturedBody["max_tokens"] != float64(100) {
		t.Errorf("expected max_tokens 100, got %v", capturedBody["max_tokens"])
	}
}

func TestChatModel_MessageConversion(t *testing.T) {
	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{"Hi"}, "stop")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewSystemMessage("You are a helpful assistant"),
			message.NewUserMessage("Hello"),
			message.NewAssistantMessage("Hi there!"),
			message.NewUserMessage("How are you?"),
		},
	})

	// Consume events
	for range events {
	}

	messages := capturedBody["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}

	// Check roles
	if messages[0].(map[string]any)["role"] != "system" {
		t.Error("expected first message role 'system'")
	}
	if messages[1].(map[string]any)["role"] != "user" {
		t.Error("expected second message role 'user'")
	}
	if messages[2].(map[string]any)["role"] != "assistant" {
		t.Error("expected third message role 'assistant'")
	}
	if messages[3].(map[string]any)["role"] != "user" {
		t.Error("expected fourth message role 'user'")
	}
}

func TestChatModel_OrganizationHeader(t *testing.T) {
	var capturedOrgHeader string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedOrgHeader = r.Header.Get("OpenAI-Organization")

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{"Hi"}, "stop")))
	}))
	defer server.Close()

	p := New(provider.Options{
		APIKey:       "test-api-key",
		BaseURL:      server.URL,
		Organization: "org-test123",
	})
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})

	// Consume events
	for range events {
	}

	if capturedOrgHeader != "org-test123" {
		t.Errorf("expected Organization header 'org-test123', got '%s'", capturedOrgHeader)
	}
}

func TestChatModel_FinishReasonLength(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{"Hello..."}, "length")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})

	var finishReason stream.FinishReason
	for event := range events {
		if e, ok := event.Data.(stream.FinishEvent); ok {
			finishReason = e.FinishReason
		}
	}

	if finishReason != stream.FinishReasonLength {
		t.Errorf("expected finish_reason 'length', got '%s'", finishReason)
	}
}

func TestChatModel_FinishReasonContentFilter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createStreamChunks([]string{""}, "content_filter")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Chat("gpt-3.5-turbo")

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
	})

	var finishReason stream.FinishReason
	for event := range events {
		if e, ok := event.Data.(stream.FinishEvent); ok {
			finishReason = e.FinishReason
		}
	}

	if finishReason != stream.FinishReasonContentFilter {
		t.Errorf("expected finish_reason 'content_filter', got '%s'", finishReason)
	}
}

// Early stream error handling (ai-sdk #15922): OpenAI can return HTTP 200 and
// then emit a top-level error frame before any output. That pre-output error
// must surface (and terminate) rather than be silently skipped.
func TestChatModel_EarlyStreamError(t *testing.T) {
	t.Run("surfaces error frame before any output", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `data: {"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		defer server.Close()

		model := createTestProvider(server.URL).Chat("gpt-4o")
		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
		})
		if err != nil {
			t.Fatal(err)
		}

		var gotErr error
		for ev := range events {
			if e, ok := ev.Data.(stream.ErrorEvent); ok {
				gotErr = e.Error
			}
		}
		if gotErr == nil {
			t.Fatal("expected an error event, got none")
		}
		if !strings.Contains(gotErr.Error(), "insufficient_quota") {
			t.Errorf("error = %v, want it to mention insufficient_quota", gotErr)
		}
	})
}
