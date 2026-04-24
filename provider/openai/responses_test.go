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
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Test fixtures for Responses API

func createResponsesStreamChunks(text string, finishReason string) string {
	var result strings.Builder

	// response.created
	createdChunk := map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":         "resp_test123",
			"created_at": 1711115037,
			"model":      "gpt-4o",
		},
	}
	jsonData, _ := json.Marshal(createdChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.output_item.added (message)
	itemAddedChunk := map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type": "message",
			"id":   "msg_test123",
			"role": "assistant",
		},
	}
	jsonData, _ = json.Marshal(itemAddedChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.output_text.delta chunks
	for _, char := range text {
		deltaChunk := map[string]any{
			"type":    "response.output_text.delta",
			"item_id": "msg_test123",
			"delta":   string(char),
		}
		jsonData, _ = json.Marshal(deltaChunk)
		fmt.Fprintf(&result, "data: %s\n\n", jsonData)
	}

	// response.output_item.done (message)
	itemDoneChunk := map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type": "message",
			"id":   "msg_test123",
			"role": "assistant",
		},
	}
	jsonData, _ = json.Marshal(itemDoneChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.completed
	completedChunk := map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":         "resp_test123",
			"created_at": 1711115037,
			"model":      "gpt-4o",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 20,
			},
		},
	}
	if finishReason != "" && finishReason != "stop" {
		completedChunk["response"].(map[string]any)["incomplete_details"] = map[string]any{
			"reason": finishReason,
		}
	}
	jsonData, _ = json.Marshal(completedChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	result.WriteString("data: [DONE]\n\n")
	return result.String()
}

func createResponsesToolCallChunks() string {
	var result strings.Builder

	// response.created
	createdChunk := map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":         "resp_test123",
			"created_at": 1711115037,
			"model":      "gpt-4o",
		},
	}
	jsonData, _ := json.Marshal(createdChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.output_item.added (function_call)
	itemAddedChunk := map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type":      "function_call",
			"id":        "fc_test123",
			"call_id":   "call_abc123",
			"name":      "get_weather",
			"arguments": "",
		},
	}
	jsonData, _ = json.Marshal(itemAddedChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.function_call_arguments.delta chunks
	argParts := []string{`{"location":`, `"San Francisco"}`}
	for _, part := range argParts {
		deltaChunk := map[string]any{
			"type":         "response.function_call_arguments.delta",
			"item_id":      "fc_test123",
			"output_index": 0,
			"delta":        part,
		}
		jsonData, _ = json.Marshal(deltaChunk)
		fmt.Fprintf(&result, "data: %s\n\n", jsonData)
	}

	// response.output_item.done (function_call)
	itemDoneChunk := map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":      "function_call",
			"id":        "fc_test123",
			"call_id":   "call_abc123",
			"name":      "get_weather",
			"arguments": `{"location":"San Francisco"}`,
			"status":    "completed",
		},
	}
	jsonData, _ = json.Marshal(itemDoneChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	// response.completed
	completedChunk := map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":         "resp_test123",
			"created_at": 1711115037,
			"model":      "gpt-4o",
			"usage": map[string]any{
				"input_tokens":  15,
				"output_tokens": 25,
			},
		},
	}
	jsonData, _ = json.Marshal(completedChunk)
	fmt.Fprintf(&result, "data: %s\n\n", jsonData)

	result.WriteString("data: [DONE]\n\n")
	return result.String()
}

// Tests

func TestResponsesModel_ID(t *testing.T) {
	p := createTestProvider("http://localhost")
	model := p.Responses("gpt-4o")

	if model.ID() != "gpt-4o" {
		t.Errorf("expected model ID 'gpt-4o', got '%s'", model.ID())
	}
}

func TestResponsesModel_Provider(t *testing.T) {
	p := createTestProvider("http://localhost")
	model := p.Responses("gpt-4o").(*ResponsesModel)

	if model.Provider() != "openai.responses" {
		t.Errorf("expected provider 'openai.responses', got '%s'", model.Provider())
	}
}

func TestResponsesModel_DefaultModel(t *testing.T) {
	p := createTestProvider("http://localhost")

	// Model() should return Responses model by default
	model := p.Model("gpt-4o")
	if _, ok := model.(*ResponsesModel); !ok {
		t.Error("expected Model() to return ResponsesModel")
	}
}

func TestResponsesModel_StreamText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.URL.Path != "/responses" {
			t.Errorf("expected path '/responses', got '%s'", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header 'Bearer test-api-key', got '%s'", r.Header.Get("Authorization"))
		}

		// Verify request body
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		json.Unmarshal(body, &reqBody)

		if reqBody["model"] != "gpt-4o" {
			t.Errorf("expected model 'gpt-4o', got '%v'", reqBody["model"])
		}
		if reqBody["stream"] != true {
			t.Errorf("expected stream true, got '%v'", reqBody["stream"])
		}
		// Check that input array exists
		if _, ok := reqBody["input"].([]any); !ok {
			t.Errorf("expected 'input' to be an array")
		}

		// Send streaming response
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesStreamChunks("Hello!", "")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

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
	if fullText != "Hello!" {
		t.Errorf("expected text 'Hello!', got '%s'", fullText)
	}

	// Verify usage
	if usage.InputTotal() != 10 {
		t.Errorf("expected prompt_tokens 10, got %d", usage.InputTotal())
	}
	if usage.OutputTotal() != 20 {
		t.Errorf("expected completion_tokens 20, got %d", usage.OutputTotal())
	}

	// Verify finish reason
	if finishReason != stream.FinishReasonStop {
		t.Errorf("expected finish_reason 'stop', got '%s'", finishReason)
	}
}

func TestResponsesModel_StreamWithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesToolCallChunks()))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

	// Create a tool (without execute function)
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

func TestResponsesModel_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"Invalid request","type":"invalid_request_error","code":"invalid_api_key"}}`))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

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

func TestResponsesModel_InputFormat(t *testing.T) {
	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesStreamChunks("Hi", "")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewSystemMessage("You are a helpful assistant"),
			message.NewUserMessage("Hello"),
		},
	})

	// Consume events
	for range events {
	}

	// Verify input format
	input := capturedBody["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("expected 2 input items, got %d", len(input))
	}

	// System message uses "system" role by default (matching ai-sdk for non-reasoning models)
	// Reasoning models (o1, o3, gpt-5) should pass ProviderOptions["systemMessageMode"] = "developer"
	systemItem := input[0].(map[string]any)
	if systemItem["role"] != "system" {
		t.Errorf("expected system message role 'system', got '%v'", systemItem["role"])
	}

	// User message
	userItem := input[1].(map[string]any)
	if userItem["role"] != "user" {
		t.Errorf("expected user message role 'user', got '%v'", userItem["role"])
	}
}

func TestResponsesModel_ToolsWithStrictSchema(t *testing.T) {
	var capturedBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesStreamChunks("Hi", "")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

	// Create a tool without additionalProperties
	tools := []tool.Tool{
		{
			Name:        "test_tool",
			Description: "A test tool",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}

	ctx := context.Background()
	events, _ := model.Stream(ctx, &stream.CallOptions{
		Messages: []message.Message{
			message.NewUserMessage("Hello"),
		},
		Tools: tools,
		ProviderOptions: map[string]any{
			"strictJsonSchema": true, // Enable strict mode for this test
		},
	})

	// Consume events
	for range events {
	}

	// Verify tools have strict mode and additionalProperties (when strictJsonSchema is true)
	toolsArr := capturedBody["tools"].([]any)
	if len(toolsArr) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(toolsArr))
	}

	toolDef := toolsArr[0].(map[string]any)
	if toolDef["strict"] != true {
		t.Error("expected strict mode to be true")
	}

	params := toolDef["parameters"].(map[string]any)
	if params["additionalProperties"] != false {
		t.Error("expected additionalProperties to be false for strict mode")
	}
}

func TestResponsesModel_FinishReasonLength(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesStreamChunks("Truncated...", "max_output_tokens")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

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

func TestResponsesModel_FinishReasonContentFilter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(createResponsesStreamChunks("", "content_filter")))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	model := p.Responses("gpt-4o")

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

// Tests for ProviderOptions - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/responses/openai-responses-language-model.test.ts

func TestResponsesModel_ReasoningEffort(t *testing.T) {
	t.Run("should send reasoningEffort provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("o3")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"reasoningEffort": "low",
			},
		})

		// Consume events
		for range events {
		}

		reasoning, ok := capturedBody["reasoning"].(map[string]any)
		if !ok {
			t.Fatal("expected reasoning object in request")
		}
		if reasoning["effort"] != "low" {
			t.Errorf("expected effort 'low', got '%v'", reasoning["effort"])
		}
	})

	t.Run("should send reasoningEffort and reasoningSummary", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("o3")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"reasoningEffort":  "medium",
				"reasoningSummary": "auto",
			},
		})

		// Consume events
		for range events {
		}

		reasoning, ok := capturedBody["reasoning"].(map[string]any)
		if !ok {
			t.Fatal("expected reasoning object in request")
		}
		if reasoning["effort"] != "medium" {
			t.Errorf("expected effort 'medium', got '%v'", reasoning["effort"])
		}
		if reasoning["summary"] != "auto" {
			t.Errorf("expected summary 'auto', got '%v'", reasoning["summary"])
		}
	})

	t.Run("should not include reasoning when not specified", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Consume events
		for range events {
		}

		if _, ok := capturedBody["reasoning"]; ok {
			t.Error("expected no reasoning object when not specified")
		}
	})
}

func TestResponsesModel_SystemMessageMode(t *testing.T) {
	t.Run("should use system role by default", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("You are helpful"),
				message.NewUserMessage("Hello"),
			},
		})

		// Consume events
		for range events {
		}

		inputArr := capturedBody["input"].([]any)
		firstMsg := inputArr[0].(map[string]any)
		if firstMsg["role"] != "system" {
			t.Errorf("expected role 'system', got '%v'", firstMsg["role"])
		}
	})

	t.Run("should use developer role when systemMessageMode is developer", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("o3")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("You are helpful"),
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"systemMessageMode": "developer",
			},
		})

		// Consume events
		for range events {
		}

		inputArr := capturedBody["input"].([]any)
		firstMsg := inputArr[0].(map[string]any)
		if firstMsg["role"] != "developer" {
			t.Errorf("expected role 'developer', got '%v'", firstMsg["role"])
		}
	})

	t.Run("should remove system messages when systemMessageMode is remove", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewSystemMessage("You are helpful"),
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"systemMessageMode": "remove",
			},
		})

		// Consume events
		for range events {
		}

		inputArr := capturedBody["input"].([]any)
		if len(inputArr) != 1 {
			t.Fatalf("expected 1 message (system removed), got %d", len(inputArr))
		}
		firstMsg := inputArr[0].(map[string]any)
		if firstMsg["role"] != "user" {
			t.Errorf("expected role 'user', got '%v'", firstMsg["role"])
		}
	})
}

func TestResponsesModel_ToolChoice(t *testing.T) {
	t.Run("should include toolChoice auto from Input", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		tools := []tool.Tool{
			{
				Name:        "test_tool",
				Description: "A test tool",
				InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
			},
		}

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Tools:      tools,
			ToolChoice: "auto",
		})

		// Consume events
		for range events {
		}

		if capturedBody["tool_choice"] != "auto" {
			t.Errorf("expected tool_choice 'auto', got '%v'", capturedBody["tool_choice"])
		}
	})

	t.Run("should include toolChoice required", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ToolChoice: "required",
		})

		// Consume events
		for range events {
		}

		if capturedBody["tool_choice"] != "required" {
			t.Errorf("expected tool_choice 'required', got '%v'", capturedBody["tool_choice"])
		}
	})

	t.Run("should not include toolChoice when nil", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		// Consume events
		for range events {
		}

		if _, ok := capturedBody["tool_choice"]; ok {
			t.Error("expected no tool_choice when not specified")
		}
	})
}

func TestResponsesModel_Store(t *testing.T) {
	t.Run("should set store to false", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"store": false,
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["store"] != false {
			t.Errorf("expected store false, got '%v'", capturedBody["store"])
		}
	})

	// Regression for ai-sdk #f4a734a: when store=false, reasoning parts
	// without encrypted_content can't round-trip and OpenAI rejects them.
	// The responses builder must drop them before sending.
	t.Run("should drop reasoning parts without encrypted_content when store=false", func(t *testing.T) {
		var capturedBody map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		events, _ := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				{
					Role: message.RoleAssistant,
					Content: message.Content{
						Parts: []message.Part{
							// Plain reasoning — no encrypted_content, should be dropped
							message.ReasoningPart{
								Text: "thinking",
								ProviderOptions: map[string]any{"itemId": "rs_1"},
							},
							// Reasoning WITH encrypted_content — should survive
							message.ReasoningPart{
								Text: "protected",
								ProviderOptions: map[string]any{
									"itemId":                    "rs_2",
									"reasoningEncryptedContent": "enc-blob",
								},
							},
							message.TextPart{Text: "final"},
						},
					},
				},
			},
			ProviderOptions: map[string]any{"store": false},
		})
		for range events {
		}

		input, _ := capturedBody["input"].([]any)
		var reasoningItems []map[string]any
		for _, item := range input {
			if m, ok := item.(map[string]any); ok && m["type"] == "reasoning" {
				reasoningItems = append(reasoningItems, m)
			}
		}
		if len(reasoningItems) != 1 {
			t.Fatalf("expected 1 reasoning item (the encrypted one), got %d", len(reasoningItems))
		}
		if reasoningItems[0]["encrypted_content"] != "enc-blob" {
			t.Errorf("wrong reasoning item kept: %v", reasoningItems[0])
		}
	})

	t.Run("should set store to true", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"store": true,
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["store"] != true {
			t.Errorf("expected store true, got '%v'", capturedBody["store"])
		}
	})
}

func TestResponsesModel_PromptCacheKey(t *testing.T) {
	t.Run("should include promptCacheKey", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"promptCacheKey": "session_123",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["prompt_cache_key"] != "session_123" {
			t.Errorf("expected prompt_cache_key 'session_123', got '%v'", capturedBody["prompt_cache_key"])
		}
	})
}

func TestResponsesModel_Include(t *testing.T) {
	t.Run("should send include provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("o3-mini")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"include": []string{"reasoning.encrypted_content"},
			},
		})

		// Consume events
		for range events {
		}

		includeArr, ok := capturedBody["include"].([]any)
		if !ok {
			t.Fatal("expected include array in request")
		}
		if len(includeArr) != 1 || includeArr[0] != "reasoning.encrypted_content" {
			t.Errorf("expected include ['reasoning.encrypted_content'], got %v", includeArr)
		}
	})

	t.Run("should send include provider option with multiple values", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("o3-mini")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"include": []string{"reasoning.encrypted_content", "file_search_call.results"},
			},
		})

		// Consume events
		for range events {
		}

		includeArr, ok := capturedBody["include"].([]any)
		if !ok {
			t.Fatal("expected include array in request")
		}
		if len(includeArr) != 2 {
			t.Errorf("expected 2 include values, got %d", len(includeArr))
		}
	})
}

func TestResponsesModel_User(t *testing.T) {
	t.Run("should send user provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"user": "user_123",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["user"] != "user_123" {
			t.Errorf("expected user 'user_123', got '%v'", capturedBody["user"])
		}
	})
}

func TestResponsesModel_ParallelToolCalls(t *testing.T) {
	t.Run("should send parallelToolCalls provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"parallelToolCalls": false,
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["parallel_tool_calls"] != false {
			t.Errorf("expected parallel_tool_calls false, got '%v'", capturedBody["parallel_tool_calls"])
		}
	})
}

func TestResponsesModel_Metadata(t *testing.T) {
	t.Run("should send metadata provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		metadata := map[string]any{
			"user_id": "user_123",
			"session": "abc",
		}

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"metadata": metadata,
			},
		})

		// Consume events
		for range events {
		}

		meta, ok := capturedBody["metadata"].(map[string]any)
		if !ok {
			t.Fatal("expected metadata object in request")
		}
		if meta["user_id"] != "user_123" {
			t.Errorf("expected metadata.user_id 'user_123', got '%v'", meta["user_id"])
		}
	})
}

func TestResponsesModel_TextVerbosity(t *testing.T) {
	t.Run("should send textVerbosity provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-5")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"textVerbosity": "low",
			},
		})

		// Consume events
		for range events {
		}

		text, ok := capturedBody["text"].(map[string]any)
		if !ok {
			t.Fatal("expected text object in request")
		}
		if text["verbosity"] != "low" {
			t.Errorf("expected text.verbosity 'low', got '%v'", text["verbosity"])
		}
	})
}

func TestResponsesModel_Truncation(t *testing.T) {
	t.Run("should send truncation auto provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"truncation": "auto",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["truncation"] != "auto" {
			t.Errorf("expected truncation 'auto', got '%v'", capturedBody["truncation"])
		}
	})

	t.Run("should send truncation disabled provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"truncation": "disabled",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["truncation"] != "disabled" {
			t.Errorf("expected truncation 'disabled', got '%v'", capturedBody["truncation"])
		}
	})
}

func TestResponsesModel_PromptCacheRetention(t *testing.T) {
	t.Run("should send promptCacheRetention provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-5.1")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"promptCacheRetention": "24h",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["prompt_cache_retention"] != "24h" {
			t.Errorf("expected prompt_cache_retention '24h', got '%v'", capturedBody["prompt_cache_retention"])
		}
	})
}

func TestResponsesModel_SafetyIdentifier(t *testing.T) {
	t.Run("should send safetyIdentifier provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"safetyIdentifier": "safety_123",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["safety_identifier"] != "safety_123" {
			t.Errorf("expected safety_identifier 'safety_123', got '%v'", capturedBody["safety_identifier"])
		}
	})
}

func TestResponsesModel_ServiceTier(t *testing.T) {
	t.Run("should send serviceTier provider option", func(t *testing.T) {
		var capturedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &capturedBody)

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(createResponsesStreamChunks("Hi", "")))
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")

		ctx := context.Background()
		events, _ := model.Stream(ctx, &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"serviceTier": "flex",
			},
		})

		// Consume events
		for range events {
		}

		if capturedBody["service_tier"] != "flex" {
			t.Errorf("expected service_tier 'flex', got '%v'", capturedBody["service_tier"])
		}
	})
}

// Exercises ai-sdk #bcb04df: response.failed should close the stream
// with a finish reason (derived from incomplete_details.reason when
// present, FinishReasonError when absent) and expose the raw reason
// via providerMetadata.openai.rawFinishReason so callers can
// disambiguate classified stop-reason errors from transport errors.
func TestResponsesModel_ResponseFailed(t *testing.T) {
	t.Run("with incomplete_details.reason maps through classifier", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			chunks := []string{
				`{"type":"response.created","response":{"id":"r","model":"gpt-4o"}}`,
				`{"type":"response.failed","response":{"id":"r","usage":{"input_tokens":5,"output_tokens":0},"incomplete_details":{"reason":"content_filter"}}}`,
			}
			for _, c := range chunks {
				fmt.Fprintf(w, "data: %s\n\n", c)
			}
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")
		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
		})
		if err != nil {
			t.Fatal(err)
		}

		var finish stream.FinishEvent
		for ev := range events {
			if f, ok := ev.Data.(stream.FinishEvent); ok {
				finish = f
			}
		}

		if finish.FinishReason != stream.FinishReasonContentFilter {
			t.Errorf("FinishReason = %q, want content-filter", finish.FinishReason)
		}
		meta, _ := finish.ProviderMetadata["openai"].(map[string]any)
		if raw, _ := meta["rawFinishReason"].(string); raw != "content_filter" {
			t.Errorf("rawFinishReason = %v, want content_filter", meta["rawFinishReason"])
		}
	})

	t.Run("without incomplete_details falls back to error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			chunks := []string{
				`{"type":"response.created","response":{"id":"r","model":"gpt-4o"}}`,
				`{"type":"response.failed","response":{"id":"r"}}`,
			}
			for _, c := range chunks {
				fmt.Fprintf(w, "data: %s\n\n", c)
			}
		}))
		defer server.Close()

		p := createTestProvider(server.URL)
		model := p.Responses("gpt-4o")
		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
		})
		if err != nil {
			t.Fatal(err)
		}

		var finish stream.FinishEvent
		for ev := range events {
			if f, ok := ev.Data.(stream.FinishEvent); ok {
				finish = f
			}
		}

		if finish.FinishReason != stream.FinishReasonError {
			t.Errorf("FinishReason = %q, want error", finish.FinishReason)
		}
		meta, _ := finish.ProviderMetadata["openai"].(map[string]any)
		if raw, _ := meta["rawFinishReason"].(string); raw != "error" {
			t.Errorf("rawFinishReason = %v, want \"error\"", meta["rawFinishReason"])
		}
	})
}
