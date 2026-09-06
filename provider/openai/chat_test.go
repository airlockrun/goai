package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

type failingStreamReader struct {
	data string
	err  error
	sent bool
}

func (r *failingStreamReader) Read(p []byte) (int, error) {
	if !r.sent {
		r.sent = true
		return copy(p, r.data), nil
	}
	return 0, r.err
}

// Test fixtures

func TestChatModel_ReasoningSettings(t *testing.T) {
	for _, tt := range []struct {
		id, effort, role    string
		sampling, reasoning bool
	}{
		{"gpt-5.6", "high", "developer", false, true},
		{"gpt-5.6", "none", "developer", true, true},
		{"gpt-6", "none", "developer", true, true},
		{"o12-mini", "none", "developer", false, true},
		{"gpt-6-chat-latest", "", "system", true, false},
		{"custom", "", "system", true, false},
	} {
		t.Run(tt.id+"/"+tt.effort, func(t *testing.T) {
			temperature, tokens := 0.4, 123
			m := &ChatModel{id: tt.id}
			body, _, err := m.buildRequest(&stream.CallOptions{Messages: []message.Message{message.NewSystemMessage("rules")}, Temperature: &temperature, TopP: &temperature, MaxOutputTokens: &tokens, Reasoning: tt.effort})
			if err != nil {
				t.Fatal(err)
			}
			var req map[string]any
			if err := json.Unmarshal(body, &req); err != nil {
				t.Fatal(err)
			}
			if (req["temperature"] != nil) != tt.sampling || (req["top_p"] != nil) != tt.sampling {
				t.Fatalf("sampling: %s", body)
			}
			if req["messages"].([]any)[0].(map[string]any)["role"] != tt.role {
				t.Fatalf("role: %s", body)
			}
			key := "max_tokens"
			if tt.reasoning {
				key = "max_completion_tokens"
			}
			if req[key] != float64(tokens) {
				t.Fatalf("tokens: %s", body)
			}
		})
	}
}

func TestChatModel_DocumentedOptions(t *testing.T) {
	zero, seed := 0.0, 7
	body, _, err := (&ChatModel{id: "gpt-4o"}).buildRequest(&stream.CallOptions{
		Seed: &seed, PresencePenalty: &zero, FrequencyPenalty: &zero,
		ProviderOptions: map[string]any{"logitBias": map[string]int{"42": 3}, "parallelToolCalls": false, "store": false, "user": "u", "serviceTier": "priority", "metadata": map[string]string{"key": "value"}, "maxCompletionTokens": 77, "safetyIdentifier": "safe", "promptCacheKey": "cache", "promptCacheRetention": "24h", "textVerbosity": "low"},
	})
	if err != nil {
		t.Fatal(err)
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"seed", "presence_penalty", "frequency_penalty", "logit_bias", "parallel_tool_calls", "store", "user", "service_tier", "metadata", "max_completion_tokens", "safety_identifier", "prompt_cache_key", "prompt_cache_retention", "verbosity"} {
		if _, ok := req[key]; !ok {
			t.Errorf("missing %s in %s", key, body)
		}
	}
}

func TestChatModel_IrregularToolIDs(t *testing.T) {
	for _, tt := range []struct{ name, chunks, wantID string }{
		{"numeric", `{"index":3,"id":42,"function":{"name":"lookup","arguments":"{}"}}`, "42"},
		{"missing", `{"index":3,"function":{"name":"lookup","arguments":"{}"}}`, ""},
		{"null", `{"index":3,"id":null,"function":{"name":"lookup","arguments":"{}"}}`, ""},
		{"delayed", `{"index":3,"function":{"arguments":"{"}}]}}]}` + "\n\ndata: " + `{"choices":[{"delta":{"tool_calls":[{"index":3,"id":"late","function":{"name":"lookup","arguments":"}"}}`, "late"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := `data: {"choices":[{"delta":{"tool_calls":[` + tt.chunks + `]}}]}` + "\n\ndata: " + `{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"
			events := make(chan stream.Event, 30)
			(&ChatModel{}).processStream(context.Background(), strings.NewReader(body), nil, events, false)
			close(events)
			var id, args string
			var calls int
			for event := range events {
				switch event.Type {
				case stream.EventError:
					t.Fatal(event.Data)
				case stream.EventToolInputStart:
					id = event.Data.(stream.ToolInputStartEvent).ID
					if id == "" {
						t.Fatal("empty start ID")
					}
				case stream.EventToolInputDelta:
					d := event.Data.(stream.ToolInputDeltaEvent)
					if d.ID != id {
						t.Fatal("delta ID mismatch")
					}
					args += d.Delta
				case stream.EventToolCall:
					c := event.Data.(stream.ToolCallEvent)
					calls++
					if c.ToolCallID != id || string(c.Input) != "{}" {
						t.Fatalf("call: %+v", c)
					}
				}
			}
			if calls != 1 || args != "{}" || tt.wantID != "" && id != tt.wantID {
				t.Fatalf("id=%q args=%q calls=%d", id, args, calls)
			}
		})
	}
}

func TestChatModel_ToolIdentityTracking(t *testing.T) {
	for _, tt := range []struct {
		name   string
		deltas []string
	}{
		{"reused index", []string{`{"index":0,"id":"a","function":{"name":"first","arguments":"{"}}`, `{"index":0,"id":"b","function":{"name":"second","arguments":"{"}}`, `{"index":0,"id":"a","function":{"arguments":"}"}}`, `{"index":0,"id":"b","function":{"arguments":"}"}}`}},
		{"no indices", []string{`{"id":"a","function":{"name":"first","arguments":"{"}}`, `{"function":{"arguments":"}"}}`, `{"id":"b","function":{"name":"second","arguments":"{"}}`, `{"id":"b","function":{"arguments":"}"}}`}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var body strings.Builder
			for _, delta := range tt.deltas {
				fmt.Fprintf(&body, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[%s]}}]}\n\n", delta)
			}
			body.WriteString("data: " + `{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n")
			events := make(chan stream.Event, 30)
			(&ChatModel{}).processStream(context.Background(), strings.NewReader(body.String()), nil, events, false)
			close(events)
			var ids []string
			for event := range events {
				if event.Type == stream.EventError {
					t.Fatal(event.Data)
				}
				if event.Type == stream.EventToolCall {
					call := event.Data.(stream.ToolCallEvent)
					ids = append(ids, call.ToolCallID)
					if string(call.Input) != "{}" {
						t.Fatalf("arguments: %s", call.Input)
					}
				}
			}
			if strings.Join(ids, ",") != "a,b" {
				t.Fatalf("calls: %v", ids)
			}
		})
	}
}

func TestChatModel_InvalidStreams(t *testing.T) {
	for _, tt := range []struct {
		name, body   string
		parse, retry bool
	}{
		{"malformed", "data: {\n\n", true, false},
		{"invalid shape", "data: {}\n\n", false, false},
		{"empty", "", false, true},
		{"truncated", "data:" + `{"choices":[{"delta":{"content":"partial"}}]}` + "\n\n", false, true},
		{"server error", "data: " + `{"error":{"code":"server_error","message":"failed"}}` + "\n\n", false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			events := make(chan stream.Event, 20)
			(&ChatModel{}).processStream(context.Background(), strings.NewReader(tt.body), nil, events, false)
			close(events)
			var got error
			for event := range events {
				if event.Type == stream.EventError {
					got = event.Data.(stream.ErrorEvent).Error
				}
				if event.Type == stream.EventFinish {
					t.Fatal("unexpected finish")
				}
			}
			if got == nil {
				t.Fatal("missing error")
			}
			var parseErr *goaierrors.JSONParseError
			if tt.parse && !errors.As(got, &parseErr) {
				t.Fatalf("not a parse error: %v", got)
			}
			var apiErr *goaierrors.APICallError
			if tt.retry && (!errors.As(got, &apiErr) || !apiErr.IsRetryable) {
				t.Fatalf("not retryable: %v", got)
			}
		})
	}
}

func createTestProvider(baseURL string) *Provider {
	return New(provider.Options{
		APIKey:  "test-api-key",
		BaseURL: baseURL,
	})
}

func TestChatModel_ProcessStreamReadError(t *testing.T) {
	readErr := errors.New("connection reset")
	events := make(chan stream.Event, 10)
	body := &failingStreamReader{
		data: `data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n",
		err:  readErr,
	}

	var model ChatModel
	model.processStream(context.Background(), body, nil, events, false)
	close(events)

	var sawError, sawFinish bool
	for event := range events {
		switch event.Type {
		case stream.EventError:
			sawError = true
			eventErr := event.Data.(stream.ErrorEvent).Error
			if !errors.Is(eventErr, readErr) {
				t.Fatalf("expected read error, got %v", eventErr)
			}
			var apiErr *goaierrors.APICallError
			if !errors.As(eventErr, &apiErr) || !apiErr.IsRetryable {
				t.Fatalf("error = %v, want retryable APICallError", eventErr)
			}
		case stream.EventFinish, stream.EventFinishStep:
			sawFinish = true
		}
	}

	if !sawError {
		t.Fatal("expected error event")
	}
	if sawFinish {
		t.Fatal("did not expect finish event after read error")
	}
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
