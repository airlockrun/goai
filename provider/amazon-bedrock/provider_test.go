package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider/anthropic"
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

type roundTripFunc func(*http.Request) (*http.Response, error)

func TestBedrockHostedToolReplayOrder(t *testing.T) {
	m := New(Options{}).Model("us.anthropic.claude-sonnet-4-6").(*BedrockLanguageModel)
	sse := ""
	for _, chunk := range []string{
		`{"type":"message_start","message":{"id":"msg"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv","name":"tool_search_tool_regex","input":{"pattern":"weather"}}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srv","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"weather"}]}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":"Found weather"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	} {
		sse += "data: " + chunk + "\n\n"
	}
	events := make(chan stream.Event, 30)
	m.processAnthropicStream(context.Background(), strings.NewReader(sse), nil, events, false, false)
	close(events)
	var parts []message.Part
	for event := range events {
		switch e := event.Data.(type) {
		case stream.ToolCallEvent:
			parts = append(parts, message.ToolCallPart{ID: e.ToolCallID, Name: e.ToolName, Input: e.Input, ProviderExecuted: e.ProviderExecuted, ProviderOptions: e.ProviderMetadata})
		case stream.ToolResultEvent:
			parts = append(parts, message.ToolResultPart{ToolCallID: e.ToolCallID, ToolName: e.ToolName, Output: e.Output, ProviderExecuted: e.ProviderExecuted, ProviderOptions: e.ProviderMetadata})
		case stream.TextDeltaEvent:
			parts = append(parts, message.TextPart{Text: e.Text})
		case stream.ErrorEvent:
			t.Fatal(e.Error)
		}
	}
	msg := message.NewAssistantMessageWithParts(parts...)
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatal(err)
	}
	body, _, err := m.buildAnthropicRequest(&stream.CallOptions{Messages: []message.Message{msg}})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []struct{ Content []map[string]any }
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	content := wire.Messages[0].Content
	if len(content) != 3 || content[0]["type"] != "server_tool_use" || content[1]["type"] != "tool_search_tool_result" || content[2]["text"] != "Found weather" || !strings.Contains(string(body), `"tool_name":"weather"`) {
		t.Fatalf("replay = %s", body)
	}
}

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestBedrockProvider_ID(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		Region:          "us-east-1",
	})

	if provider.ID() != "amazon-bedrock" {
		t.Errorf("expected provider ID amazon-bedrock, got %s", provider.ID())
	}
}

func TestBedrockLanguageModel_StreamReadErrors(t *testing.T) {
	readErr := errors.New("connection reset")
	model := &BedrockLanguageModel{}
	tests := []struct {
		name string
		data string
		run  func(io.Reader, chan<- stream.Event)
	}{
		{
			name: "anthropic parser",
			data: `data: {"type":"content_block_start","content_block":{"type":"text"}}` + "\n\n",
			run: func(reader io.Reader, events chan<- stream.Event) {
				model.processAnthropicStream(context.Background(), reader, nil, events, false, false)
			},
		},
		{
			name: "generic parser",
			data: `{"outputText":"partial"}` + "\n",
			run: func(reader io.Reader, events chan<- stream.Event) {
				model.processGenericStream(context.Background(), reader, events, false)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := make(chan stream.Event, 10)
			tt.run(&failingStreamReader{data: tt.data, err: readErr}, events)
			close(events)

			var eventErr error
			for event := range events {
				if event.Type == stream.EventError {
					eventErr = event.Data.(stream.ErrorEvent).Error
				}
				if event.Type == stream.EventFinish || event.Type == stream.EventFinishStep {
					t.Fatal("unexpected finish after stream read error")
				}
			}
			var apiErr *goaierrors.APICallError
			if !errors.Is(eventErr, readErr) || !errors.As(eventErr, &apiErr) || !apiErr.IsRetryable {
				t.Fatalf("error = %v, want retryable APICallError wrapping read error", eventErr)
			}
		})
	}
}

func TestBedrockLanguageModel_SetupErrorClassification(t *testing.T) {
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
			model := New(Options{AccessKeyID: "key", SecretAccessKey: "secret"}).Model("anthropic.claude-3-haiku")
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

func TestBedrockLanguageModel_ID(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")

	if m.ID() != "anthropic.claude-3-5-sonnet-20241022-v2:0" {
		t.Errorf("expected model ID anthropic.claude-3-5-sonnet-20241022-v2:0, got %s", m.ID())
	}
}

func TestBedrockLanguageModel_Provider(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")

	if m.Provider() != "amazon-bedrock" {
		t.Errorf("expected provider amazon-bedrock, got %s", m.Provider())
	}
}

func TestBedrockLanguageModel_Stream(t *testing.T) {
	t.Run("should create anthropic model correctly", func(t *testing.T) {
		// Note: Bedrock uses AWS signing and constructs URLs dynamically,
		// which makes it difficult to mock completely.
		// This test verifies the provider structure.

		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
			Region:          "us-east-1",
		})

		m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})

	t.Run("should build correct anthropic request", func(t *testing.T) {
		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
			Region:          "us-east-1",
		})

		m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")
		langModel := m.(*BedrockLanguageModel)

		// Test building the request
		maxTokens := 1000
		temp := 0.7
		input := &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleSystem, Content: message.Content{Text: "You are helpful"}},
				{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
			},
			MaxOutputTokens: &maxTokens,
			Temperature:     &temp,
		}

		reqBytes, _, err := langModel.buildAnthropicRequest(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var reqBody map[string]any
		json.Unmarshal(reqBytes, &reqBody)

		if reqBody["anthropic_version"] != "bedrock-2023-05-31" {
			t.Errorf("expected anthropic_version 'bedrock-2023-05-31', got %v", reqBody["anthropic_version"])
		}

		if reqBody["system"] != "You are helpful" {
			t.Errorf("expected system 'You are helpful', got %v", reqBody["system"])
		}

		if reqBody["max_tokens"] != float64(1000) {
			t.Errorf("expected max_tokens 1000, got %v", reqBody["max_tokens"])
		}
	})

	t.Run("should build correct titan request", func(t *testing.T) {
		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
		})

		m := provider.LanguageModel("amazon.titan-text-express-v1")
		langModel := m.(*BedrockLanguageModel)

		input := &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
			},
		}

		reqBytes, _, err := langModel.buildTitanRequest(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var reqBody map[string]any
		json.Unmarshal(reqBytes, &reqBody)

		if reqBody["inputText"] == nil {
			t.Error("expected inputText in request")
		}
	})

	t.Run("should build correct llama request", func(t *testing.T) {
		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
		})

		m := provider.LanguageModel("meta.llama3-8b-instruct-v1:0")
		langModel := m.(*BedrockLanguageModel)

		input := &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
			},
		}

		reqBytes, _, err := langModel.buildLlamaRequest(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var reqBody map[string]any
		json.Unmarshal(reqBytes, &reqBody)

		if reqBody["prompt"] == nil {
			t.Error("expected prompt in request")
		}
	})

	t.Run("should build correct mistral request", func(t *testing.T) {
		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
		})

		m := provider.LanguageModel("mistral.mistral-7b-instruct-v0:2")
		langModel := m.(*BedrockLanguageModel)

		input := &stream.CallOptions{
			Messages: []message.Message{
				{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
			},
		}

		reqBytes, _, err := langModel.buildMistralRequest(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var reqBody map[string]any
		json.Unmarshal(reqBytes, &reqBody)

		if reqBody["prompt"] == nil {
			t.Error("expected prompt in request")
		}
	})
}

func TestBedrockProvider_DefaultRegion(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})

	// Region should default to us-east-1
	if provider.opts.Region != "us-east-1" {
		t.Errorf("expected default region us-east-1, got %s", provider.opts.Region)
	}
}

func TestBedrockProvider_BaseURL(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		Region:          "us-west-2",
	})

	baseURL := provider.baseURL()
	expected := "https://bedrock-runtime.us-west-2.amazonaws.com"

	if baseURL != expected {
		t.Errorf("expected base URL %s, got %s", expected, baseURL)
	}
}

func TestBedrockImageModel_ID(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.ImageModel("amazon.titan-image-generator-v1")

	if m.ID() != "amazon.titan-image-generator-v1" {
		t.Errorf("expected model ID amazon.titan-image-generator-v1, got %s", m.ID())
	}
}

func TestBedrockImageModel_Provider(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.ImageModel("amazon.titan-image-generator-v1")

	if m.Provider() != "amazon-bedrock" {
		t.Errorf("expected provider amazon-bedrock, got %s", m.Provider())
	}
}

func TestBedrockEmbeddingModel_ID(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.EmbeddingModel("amazon.titan-embed-text-v1")

	if m.ID() != "amazon.titan-embed-text-v1" {
		t.Errorf("expected model ID amazon.titan-embed-text-v1, got %s", m.ID())
	}
}

func TestBedrockEmbeddingModel_Provider(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	})
	m := provider.EmbeddingModel("amazon.titan-embed-text-v1")

	if m.Provider() != "amazon-bedrock" {
		t.Errorf("expected provider amazon-bedrock, got %s", m.Provider())
	}
}

func TestBedrockLanguageModel_ErrorHandling(t *testing.T) {
	t.Run("model should be created correctly", func(t *testing.T) {
		provider := New(Options{
			AccessKeyID:     "test-access-key",
			SecretAccessKey: "test-secret-key",
		})

		m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
	})
}

func TestBedrockAWSSignature(t *testing.T) {
	t.Run("should generate valid SHA256 hash", func(t *testing.T) {
		hash := sha256Hash([]byte("test data"))
		if len(hash) != 64 {
			t.Errorf("expected 64 character hex string, got %d characters", len(hash))
		}
	})

	t.Run("should generate valid HMAC", func(t *testing.T) {
		result := hmacSHA256([]byte("key"), []byte("data"))
		if len(result) != 32 {
			t.Errorf("expected 32 byte HMAC, got %d bytes", len(result))
		}
	})
}

// Tests for ProviderOptions - verifies ChatOptions are wired up correctly

func TestBedrockProviderOptions_ReasoningConfig(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		Region:          "us-east-1",
	})

	m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")
	langModel := m.(*BedrockLanguageModel)

	input := &stream.CallOptions{
		Messages: []message.Message{
			{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
		},
		ProviderOptions: map[string]any{
			"reasoningConfig": map[string]any{
				"type":               "enabled",
				"budgetTokens":       1024,
				"maxReasoningEffort": "high",
			},
		},
	}

	reqBytes, _, err := langModel.buildAnthropicRequest(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var reqBody map[string]any
	json.Unmarshal(reqBytes, &reqBody)

	thinking, ok := reqBody["thinking"].(map[string]any)
	if !ok {
		t.Fatal("expected thinking in request body")
	}

	if thinking["type"] != "enabled" {
		t.Errorf("expected type 'enabled', got %v", thinking["type"])
	}
	if thinking["budget_tokens"] != float64(1024) {
		t.Errorf("expected budget_tokens 1024, got %v", thinking["budget_tokens"])
	}
	if thinking["max_reasoning_effort"] != "high" {
		t.Errorf("expected max_reasoning_effort 'high', got %v", thinking["max_reasoning_effort"])
	}
}

func TestBedrockProviderOptions_AdditionalFields(t *testing.T) {
	provider := New(Options{
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		Region:          "us-east-1",
	})

	m := provider.LanguageModel("anthropic.claude-3-5-sonnet-20241022-v2:0")
	langModel := m.(*BedrockLanguageModel)

	input := &stream.CallOptions{
		Messages: []message.Message{
			{Role: message.RoleUser, Content: message.Content{Text: "Hello"}},
		},
		ProviderOptions: map[string]any{
			"additionalModelRequestFields": map[string]any{
				"custom_field":  "custom_value",
				"another_field": 42,
			},
		},
	}

	reqBytes, _, err := langModel.buildAnthropicRequest(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var reqBody map[string]any
	json.Unmarshal(reqBytes, &reqBody)

	if reqBody["custom_field"] != "custom_value" {
		t.Errorf("expected custom_field 'custom_value', got %v", reqBody["custom_field"])
	}
	if reqBody["another_field"] != float64(42) {
		t.Errorf("expected another_field 42, got %v", reqBody["another_field"])
	}
}

func TestBedrockModel_ResponseFormat(t *testing.T) {
	p := New(Options{
		AccessKeyID:     "k",
		SecretAccessKey: "s",
		Region:          "us-east-1",
	})

	t.Run("anthropic family with schema injects synthetic tool and tool_choice", func(t *testing.T) {
		m := p.Model("anthropic.claude-3-sonnet-20240229-v1:0").(*BedrockLanguageModel)
		schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages:       []message.Message{message.NewUserMessage("hi")},
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		tools, _ := body["tools"].([]any)
		found := false
		for _, tl := range tools {
			tm, _ := tl.(map[string]any)
			if tm["name"] == "json" {
				found = true
			}
		}
		if !found {
			t.Errorf("expected synthetic json tool, got %v", tools)
		}
		tc, _ := body["tool_choice"].(map[string]any)
		if tc["name"] != "json" {
			t.Errorf("expected tool_choice to force json, got %v", tc)
		}
	})

	t.Run("anthropic family without schema injects JSON instruction into system", func(t *testing.T) {
		m := p.Model("anthropic.claude-3-sonnet-20240229-v1:0").(*BedrockLanguageModel)
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages:       []message.Message{message.NewUserMessage("hi")},
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		sys, _ := body["system"].(string)
		if !strings.Contains(sys, "JSON") {
			t.Errorf("expected injected JSON instruction in system, got %q", sys)
		}
	})

	t.Run("titan family injects JSON instruction into prompt", func(t *testing.T) {
		m := p.Model("amazon.titan-text-express-v1").(*BedrockLanguageModel)
		schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
		raw, _, err := m.buildTitanRequest(&stream.CallOptions{
			Messages:       []message.Message{message.NewUserMessage("hi")},
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		input, _ := body["inputText"].(string)
		if !strings.Contains(input, "JSON schema:") {
			t.Errorf("expected injected JSON schema instruction, got %q", input)
		}
	})

	t.Run("mistral family wraps injected instruction as [INST] block", func(t *testing.T) {
		m := p.Model("mistral.mistral-7b-instruct-v0:2").(*BedrockLanguageModel)
		raw, _, err := m.buildMistralRequest(&stream.CallOptions{
			Messages:       []message.Message{message.NewUserMessage("hi")},
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		p, _ := body["prompt"].(string)
		if !strings.Contains(p, "JSON") {
			t.Errorf("expected injected JSON instruction, got %q", p)
		}
	})

	t.Run("llama family injects JSON instruction into system header", func(t *testing.T) {
		m := p.Model("meta.llama3-70b-instruct-v1:0").(*BedrockLanguageModel)
		raw, _, err := m.buildLlamaRequest(&stream.CallOptions{
			Messages:       []message.Message{message.NewUserMessage("hi")},
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		p, _ := body["prompt"].(string)
		if !strings.Contains(p, "JSON") {
			t.Errorf("expected injected JSON instruction, got %q", p)
		}
	})
}

// Bedrock rejects output_config.format for claude-opus-4-7, so it falls back
// to the synthetic JSON tool even when structuredOutputMode=outputFormat is
// requested. Other Anthropic models keep the native output_config.format path.
// Mirrors ai-sdk #15288.
func TestBedrockModel_StructuredOutputOpus47(t *testing.T) {
	p := New(Options{AccessKeyID: "k", SecretAccessKey: "s", Region: "us-east-1"})
	schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)

	hasOutputConfig := func(modelID string) bool {
		m := p.Model(modelID).(*BedrockLanguageModel)
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages:        []message.Message{message.NewUserMessage("hi")},
			ResponseFormat:  &stream.ResponseFormat{Type: "json", Schema: schema},
			ProviderOptions: map[string]any{"structuredOutputMode": "outputFormat"},
		})
		if err != nil {
			t.Fatalf("%s: %v", modelID, err)
		}
		var body map[string]any
		json.Unmarshal(raw, &body)
		_, ok := body["output_config"]
		return ok
	}

	tests := []struct {
		modelID       string
		wantOutputCfg bool
	}{
		{"anthropic.claude-opus-4-7", false},
		{"us.anthropic.claude-opus-4-7", false},
		{"global.anthropic.claude-opus-4-8", false},
		{"us.anthropic.claude-sonnet-5", false},
		{"eu.anthropic.claude-opus-5", false},
		{"arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-fable-5-1", false},
		{"anthropic.claude-sonnet-4-6", false},
		{"anthropic.claude-haiku-4-5", false},
		{"anthropic.claude-opus-4-6-v1", true},
	}
	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			if got := hasOutputConfig(tt.modelID); got != tt.wantOutputCfg {
				t.Errorf("output_config present = %v, want %v", got, tt.wantOutputCfg)
			}
		})
	}
}

// Exercises ai-sdk PRs #df099b9 (serviceTier), #b128d9b (cacheControl TTL),
// #91f8777 (tool strict mode), #a1a8091 (tool_choice passthrough).
func TestBedrockStrictExclusions(t *testing.T) {
	for _, tc := range []struct {
		id     string
		strict bool
	}{{"claude-sonnet-5", false}, {"claude-opus-5", false}, {"claude-fable-5-1", false}, {"claude-opus-4-7", false}, {"claude-opus-4-8", false}, {"claude-sonnet-4-6", true}, {"claude-haiku-4-5", true}, {"claude-opus-4-6", true}} {
		t.Run(tc.id, func(t *testing.T) {
			m := New(Options{}).Model("arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic." + tc.id).(*BedrockLanguageModel)
			body, _, err := m.buildAnthropicRequest(&stream.CallOptions{Tools: []tool.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}}})
			if err != nil {
				t.Fatal(err)
			}
			var req struct {
				Tools []map[string]any `json:"tools"`
			}
			if err := json.Unmarshal(body, &req); err != nil {
				t.Fatal(err)
			}
			_, strict := req.Tools[0]["strict"]
			if strict != tc.strict {
				t.Fatalf("%s", body)
			}
		})
	}
}

func TestBedrockAnthropic_RequestBodyWiring(t *testing.T) {
	p := New(Options{AccessKeyID: "k", SecretAccessKey: "s", Region: "us-east-1"})
	m := p.Model("anthropic.claude-opus-4-6-v1").(*BedrockLanguageModel)

	raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
		Tools: []tool.Tool{{
			Name:        "lookup",
			Description: "lookup something",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		ToolChoice: "auto",
		ProviderOptions: map[string]any{
			"serviceTier": "priority",
			"cacheControl": map[string]any{
				"type": "ephemeral",
				"ttl":  "1h",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}

	if body["service_tier"] != "priority" {
		t.Errorf("service_tier = %v, want priority", body["service_tier"])
	}
	cc, ok := body["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("cache_control = %v (type %T)", body["cache_control"], body["cache_control"])
	}
	if cc["ttl"] != "1h" {
		t.Errorf("cache_control.ttl = %v, want 1h", cc["ttl"])
	}
	// Anthropic-on-Bedrock uses the same Messages-API wire format as direct
	// Anthropic: tool_choice is an object, not a bare string. The provider
	// translates "auto" → {type: "auto"}.
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "auto" {
		t.Errorf("tool_choice = %v, want {type: auto}", body["tool_choice"])
	}
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	t0 := tools[0].(map[string]any)
	if t0["strict"] != true {
		t.Errorf("tools[0].strict = %v, want true", t0["strict"])
	}
}

// Exercises ai-sdk #ff854a2 — Bedrock must auto-inject the
// tool-search-tool-2025-10-19 anthropic_beta whenever a tool-search
// hosted tool is present, and must preserve user-supplied betas with
// no duplicates.
func TestBedrockAnthropic_ToolSearchBetaInjection(t *testing.T) {
	p := New(Options{AccessKeyID: "k", SecretAccessKey: "s", Region: "us-east-1"})
	m := p.Model("anthropic.claude-opus-4-6-v1").(*BedrockLanguageModel)

	t.Run("regex auto-injects beta and emits wire shape", func(t *testing.T) {
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools:    []tool.Tool{anthropic.ToolSearchRegex()},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}

		tools, ok := body["tools"].([]any)
		if !ok || len(tools) != 1 {
			t.Fatalf("tools = %v", body["tools"])
		}
		t0 := tools[0].(map[string]any)
		if t0["type"] != "tool_search_tool_regex_20251119" {
			t.Errorf("tools[0].type = %v, want tool_search_tool_regex_20251119", t0["type"])
		}
		if t0["name"] != "tool_search_tool_regex" {
			t.Errorf("tools[0].name = %v, want tool_search_tool_regex", t0["name"])
		}
		// Function-tool-only field must NOT leak onto hosted tools.
		if _, hasStrict := t0["strict"]; hasStrict {
			t.Errorf("hosted tool must not have strict field, got %v", t0)
		}

		betas, ok := body["anthropic_beta"].([]any)
		if !ok {
			t.Fatalf("anthropic_beta = %v (type %T)", body["anthropic_beta"], body["anthropic_beta"])
		}
		if len(betas) != 1 || betas[0] != "tool-search-tool-2025-10-19" {
			t.Errorf("anthropic_beta = %v, want [tool-search-tool-2025-10-19]", betas)
		}
	})

	t.Run("bm25 auto-injects beta", func(t *testing.T) {
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools:    []tool.Tool{anthropic.ToolSearchBM25()},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}

		tools, _ := body["tools"].([]any)
		t0 := tools[0].(map[string]any)
		if t0["type"] != "tool_search_tool_bm25_20251119" {
			t.Errorf("tools[0].type = %v, want tool_search_tool_bm25_20251119", t0["type"])
		}
		betas, _ := body["anthropic_beta"].([]any)
		if len(betas) != 1 || betas[0] != "tool-search-tool-2025-10-19" {
			t.Errorf("anthropic_beta = %v, want [tool-search-tool-2025-10-19]", betas)
		}
	})

	t.Run("user-supplied betas merge without duplicates", func(t *testing.T) {
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools:    []tool.Tool{anthropic.ToolSearchRegex(), anthropic.ToolSearchBM25()},
			ProviderOptions: map[string]any{
				"anthropicBeta": []any{"custom-beta-2025-01-01", "tool-search-tool-2025-10-19"},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		betas, ok := body["anthropic_beta"].([]any)
		if !ok {
			t.Fatalf("anthropic_beta = %v (type %T)", body["anthropic_beta"], body["anthropic_beta"])
		}
		// Expect: user-supplied first (order preserved), auto-injected
		// duplicate collapsed.
		want := []string{"custom-beta-2025-01-01", "tool-search-tool-2025-10-19"}
		if len(betas) != len(want) {
			t.Fatalf("anthropic_beta len = %d, want %d: %v", len(betas), len(want), betas)
		}
		for i, v := range want {
			if betas[i] != v {
				t.Errorf("anthropic_beta[%d] = %v, want %q", i, betas[i], v)
			}
		}
	})

	t.Run("no tool-search tools means no beta header", func(t *testing.T) {
		raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
			Messages: []message.Message{message.NewUserMessage("hi")},
			Tools: []tool.Tool{{
				Name:        "lookup",
				Description: "lookup",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		if _, has := body["anthropic_beta"]; has {
			t.Errorf("anthropic_beta should be absent when no betas collected, got %v", body["anthropic_beta"])
		}
	})
}

// Verifies ai-sdk@6.0.168 BEDROCK_TOOL_BETA_MAP parity: bash, text_editor,
// and computer hosted-tool wire-types each inject the matching
// anthropic_beta on Bedrock.
func TestBedrockAnthropic_ComputerUseBetaInjection(t *testing.T) {
	p := New(Options{AccessKeyID: "k", SecretAccessKey: "s", Region: "us-east-1"})
	m := p.Model("anthropic.claude-opus-4-6-v1").(*BedrockLanguageModel)

	computerOpts := anthropic.ComputerOptions{DisplayWidthPx: 800, DisplayHeightPx: 600}

	tests := []struct {
		name     string
		tool     tool.Tool
		wantBeta string
	}{
		{"bash_20241022", anthropic.Bash20241022(), "computer-use-2024-10-22"},
		{"bash_20250124", anthropic.Bash20250124(), "computer-use-2025-01-24"},
		{"text_editor_20241022", anthropic.TextEditor20241022(), "computer-use-2024-10-22"},
		{"text_editor_20250124", anthropic.TextEditor20250124(), "computer-use-2025-01-24"},
		{"text_editor_20250429", anthropic.TextEditor20250429(), "computer-use-2025-01-24"},
		{"text_editor_20250728", anthropic.TextEditor20250728(), "computer-use-2025-01-24"},
		{"computer_20241022", anthropic.Computer20241022With(computerOpts), "computer-use-2024-10-22"},
		{"computer_20250124", anthropic.Computer20250124With(computerOpts), "computer-use-2025-01-24"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, _, err := m.buildAnthropicRequest(&stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("hi")},
				Tools:    []tool.Tool{tc.tool},
			})
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatal(err)
			}
			betas, ok := body["anthropic_beta"].([]any)
			if !ok {
				t.Fatalf("anthropic_beta missing or wrong type: %v", body["anthropic_beta"])
			}
			found := false
			for _, b := range betas {
				if b == tc.wantBeta {
					found = true
				}
			}
			if !found {
				t.Errorf("anthropic_beta = %v, want to contain %q", betas, tc.wantBeta)
			}
		})
	}
}
