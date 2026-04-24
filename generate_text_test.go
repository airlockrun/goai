package goai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/output"
	"github.com/airlockrun/goai/schema"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
	"github.com/airlockrun/goai/tool"
)

// Tests for GenerateText and StreamText using MockLanguageModel
// Source: ai-sdk/packages/ai/src/generate-text/generate-text.test.ts
// Source: ai-sdk/packages/ai/src/generate-text/stream-text.test.ts

func TestGenerateText_BasicUsage(t *testing.T) {
	t.Run("should generate text", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Hello, world!", testutil.MockUsage(10, 20)),
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("Say hello"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Text != "Hello, world!" {
			t.Errorf("expected 'Hello, world!', got '%s'", result.Text)
		}
		if result.FinishReason != stream.FinishReasonStop {
			t.Errorf("expected stop finish reason, got %s", result.FinishReason)
		}
		if result.Usage.InputTotal() != 10 {
			t.Errorf("expected 10 prompt tokens, got %d", result.Usage.InputTotal())
		}
		if result.Usage.OutputTotal() != 20 {
			t.Errorf("expected 20 completion tokens, got %d", result.Usage.OutputTotal())
		}
	})

	t.Run("should record messages correctly", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("Response text", testutil.MockUsage(5, 10)),
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewSystemMessage("You are helpful"),
				message.NewUserMessage("Hello"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should have original messages + assistant response
		if len(result.Response.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(result.Response.Messages))
		}

		// Last message should be assistant
		lastMsg := result.Response.Messages[2]
		if lastMsg.Role != message.RoleAssistant {
			t.Errorf("expected assistant role, got %s", lastMsg.Role)
		}
	})

	t.Run("should capture tool calls", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockToolCallResponse(
				"call_123",
				"get_weather",
				map[string]string{"location": "NYC"},
				testutil.MockUsage(15, 25),
			),
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("What's the weather?"),
			},
			Tools: tool.Set{
				"get_weather": tool.Tool{
					Name:        "get_weather",
					Description: "Get weather",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
		}
		if result.ToolCalls[0].Name != "get_weather" {
			t.Errorf("expected tool name 'get_weather', got '%s'", result.ToolCalls[0].Name)
		}
		if result.FinishReason != stream.FinishReasonToolCalls {
			t.Errorf("expected tool-calls finish reason, got %s", result.FinishReason)
		}
	})

	t.Run("should pass model input correctly", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("test", testutil.MockUsage(1, 1)),
		})

		temp := 0.7
		topP := 0.9
		maxTokens := 100

		_, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("test"),
			},
			Temperature:     &temp,
			TopP:            &topP,
			MaxOutputTokens: &maxTokens,
			StopSequences:   []string{"END"},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(model.DoStreamCalls) != 1 {
			t.Fatalf("expected 1 call, got %d", len(model.DoStreamCalls))
		}

		input := model.DoStreamCalls[0]
		if *input.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %f", *input.Temperature)
		}
		if *input.TopP != 0.9 {
			t.Errorf("expected topP 0.9, got %f", *input.TopP)
		}
		if *input.MaxOutputTokens != 100 {
			t.Errorf("expected maxTokens 100, got %d", *input.MaxOutputTokens)
		}
		if len(input.StopSequences) != 1 || input.StopSequences[0] != "END" {
			t.Errorf("expected stop sequence [END], got %v", input.StopSequences)
		}
	})
}

func TestStreamText_BasicUsage(t *testing.T) {
	t.Run("should stream text chunks", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockStreamedTextResponse(
				[]string{"Hello", ", ", "world", "!"},
				testutil.MockUsage(10, 4),
			),
		})

		result, err := StreamText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("Say hello"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Collect all events
		var events []stream.Event
		for event := range result.FullStream {
			events = append(events, event)
		}

		// Should have: start-step, text-start, 4 text-deltas, text-end, finish-step, finish = 9 events
		// (start-step and final finish are added by multi-step loop)
		if len(events) < 7 {
			t.Fatalf("expected at least 7 events, got %d", len(events))
		}

		// Check final text
		if result.Text() != "Hello, world!" {
			t.Errorf("expected 'Hello, world!', got '%s'", result.Text())
		}
	})

	t.Run("should provide usage after completion", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse("test", testutil.MockUsage(100, 50)),
		})

		result, err := StreamText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("test"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range result.FullStream {
		}

		usage := result.Usage()
		if usage.InputTotal() != 100 {
			t.Errorf("expected 100 prompt tokens, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 50 {
			t.Errorf("expected 50 completion tokens, got %d", usage.OutputTotal())
		}
	})

	t.Run("should capture tool calls", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextWithToolCallResponse(
				"I'll check the weather",
				"call_456",
				"get_weather",
				map[string]string{"location": "London"},
				testutil.MockUsage(20, 30),
			),
		})

		result, err := StreamText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("What's the weather in London?"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range result.FullStream {
		}

		toolCalls := result.ToolCalls()
		if len(toolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
		}
		if toolCalls[0].Name != "get_weather" {
			t.Errorf("expected 'get_weather', got '%s'", toolCalls[0].Name)
		}
		if toolCalls[0].ID != "call_456" {
			t.Errorf("expected 'call_456', got '%s'", toolCalls[0].ID)
		}
	})
}

func TestGenerateText_WithMultipleResponses(t *testing.T) {
	t.Run("should handle sequential calls with different responses", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockTextResponse("First response", testutil.MockUsage(10, 10)),
				testutil.MockTextResponse("Second response", testutil.MockUsage(10, 15)),
			},
		})

		// First call
		result1, err := GenerateText(context.Background(), stream.Input{
			Model:    model,
			Messages: []message.Message{message.NewUserMessage("First")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result1.Text != "First response" {
			t.Errorf("expected 'First response', got '%s'", result1.Text)
		}

		// Second call
		result2, err := GenerateText(context.Background(), stream.Input{
			Model:    model,
			Messages: []message.Message{message.NewUserMessage("Second")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result2.Text != "Second response" {
			t.Errorf("expected 'Second response', got '%s'", result2.Text)
		}
	})
}

// Tests for multi-step tool loop (MaxSteps > 1)
// Source: ai-sdk/packages/ai/src/generate-text/generate-text.test.ts (options.stopWhen)

func TestGenerateText_MultiStep(t *testing.T) {
	t.Run("should execute tools and continue loop when MaxSteps > 1", func(t *testing.T) {
		// Step 1: Model returns tool call
		// Step 2: Tool executed, model returns final text
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse(
					"call_1",
					"get_weather",
					map[string]string{"location": "NYC"},
					testutil.MockUsage(10, 15),
				),
				testutil.MockTextResponse("The weather in NYC is sunny.", testutil.MockUsage(20, 10)),
			},
		})

		var stepFinishCount int
		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("What's the weather in NYC?"),
			},
			MaxSteps: 3,
			Tools: tool.Set{
				"get_weather": tool.Tool{
					Name:        "get_weather",
					Description: "Get weather for a location",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "Sunny, 72F"}, nil
					},
				},
			},
			OnStepFinish: func(step stream.StepResultData) {
				stepFinishCount++
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should have 2 steps
		if len(result.Steps) != 2 {
			t.Errorf("expected 2 steps, got %d", len(result.Steps))
		}

		// Final text should be from step 2
		if result.Text != "The weather in NYC is sunny." {
			t.Errorf("expected final text 'The weather in NYC is sunny.', got '%s'", result.Text)
		}

		// OnStepFinish should be called twice
		if stepFinishCount != 2 {
			t.Errorf("expected onStepFinish to be called 2 times, got %d", stepFinishCount)
		}

		// Check step 1 had tool call
		if len(result.Steps[0].ToolCalls()) != 1 {
			t.Errorf("step 1 should have 1 tool call, got %d", len(result.Steps[0].ToolCalls()))
		}
		if result.Steps[0].FinishReason != stream.FinishReasonToolCalls {
			t.Errorf("step 1 finish reason should be tool-calls, got %s", result.Steps[0].FinishReason)
		}

		// Check step 2 had no tool calls
		if len(result.Steps[1].ToolCalls()) != 0 {
			t.Errorf("step 2 should have 0 tool calls, got %d", len(result.Steps[1].ToolCalls()))
		}
		if result.Steps[1].FinishReason != stream.FinishReasonStop {
			t.Errorf("step 2 finish reason should be stop, got %s", result.Steps[1].FinishReason)
		}

		// Check total usage is accumulated
		expectedPromptTokens := 30     // 10 + 20
		expectedCompletionTokens := 25 // 15 + 10
		if result.Usage.InputTotal() != expectedPromptTokens {
			t.Errorf("expected %d prompt tokens, got %d", expectedPromptTokens, result.Usage.InputTotal())
		}
		if result.Usage.OutputTotal() != expectedCompletionTokens {
			t.Errorf("expected %d completion tokens, got %d", expectedCompletionTokens, result.Usage.OutputTotal())
		}
	})

	t.Run("should stop at MaxSteps even if model keeps returning tool calls", func(t *testing.T) {
		// All steps return tool calls
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse("call_1", "search", map[string]string{"q": "1"}, testutil.MockUsage(10, 10)),
				testutil.MockToolCallResponse("call_2", "search", map[string]string{"q": "2"}, testutil.MockUsage(10, 10)),
				testutil.MockToolCallResponse("call_3", "search", map[string]string{"q": "3"}, testutil.MockUsage(10, 10)),
			},
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("Search"),
			},
			MaxSteps: 2, // Limit to 2 steps
			Tools: tool.Set{
				"search": tool.Tool{
					Name:        "search",
					Description: "Search",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "result"}, nil
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should stop at 2 steps
		if len(result.Steps) != 2 {
			t.Errorf("expected 2 steps, got %d", len(result.Steps))
		}

		// Model should have been called twice
		if len(model.DoStreamCalls) != 2 {
			t.Errorf("expected 2 model calls, got %d", len(model.DoStreamCalls))
		}
	})

	t.Run("should include tool results in messages for next step", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse(
					"call_1",
					"get_data",
					map[string]string{"key": "test"},
					testutil.MockUsage(10, 10),
				),
				testutil.MockTextResponse("Got the data.", testutil.MockUsage(20, 5)),
			},
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("Get data"),
			},
			MaxSteps: 3,
			Tools: tool.Set{
				"get_data": tool.Tool{
					Name:        "get_data",
					Description: "Get data",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "data value"}, nil
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Check second call had tool result message
		if len(model.DoStreamCalls) != 2 {
			t.Fatalf("expected 2 model calls, got %d", len(model.DoStreamCalls))
		}

		secondCallMessages := model.DoStreamCalls[1].Messages
		// Should have: user, assistant (with tool call), tool result
		if len(secondCallMessages) < 3 {
			t.Fatalf("expected at least 3 messages in second call, got %d", len(secondCallMessages))
		}

		// Last message should be tool role
		lastMsg := secondCallMessages[len(secondCallMessages)-1]
		if lastMsg.Role != message.RoleTool {
			t.Errorf("expected last message to be tool role, got %s", lastMsg.Role)
		}

		// Response messages should contain all messages
		if len(result.Response.Messages) < 4 {
			t.Errorf("expected at least 4 response messages, got %d", len(result.Response.Messages))
		}
	})

	t.Run("should default to single step (MaxSteps=1) when not specified", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse("call_1", "tool", map[string]string{}, testutil.MockUsage(10, 10)),
				testutil.MockTextResponse("Never reached", testutil.MockUsage(5, 5)),
			},
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("test"),
			},
			// MaxSteps not set - should default to 1
			Tools: tool.Set{
				"tool": tool.Tool{
					Name:        "tool",
					Description: "A tool",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "result"}, nil
					},
				},
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should only have 1 step
		if len(result.Steps) != 1 {
			t.Errorf("expected 1 step, got %d", len(result.Steps))
		}

		// Model should only be called once
		if len(model.DoStreamCalls) != 1 {
			t.Errorf("expected 1 model call, got %d", len(model.DoStreamCalls))
		}
	})
}

func TestStreamText_MultiStep(t *testing.T) {
	t.Run("should stream events from all steps", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse(
					"call_1",
					"get_time",
					map[string]string{},
					testutil.MockUsage(10, 10),
				),
				testutil.MockTextResponse("The time is 12:00", testutil.MockUsage(15, 8)),
			},
		})

		var stepFinishCount int
		result, err := StreamText(context.Background(), stream.Input{
			Model: model,
			Messages: []message.Message{
				message.NewUserMessage("What time is it?"),
			},
			MaxSteps: 3,
			Tools: tool.Set{
				"get_time": tool.Tool{
					Name:        "get_time",
					Description: "Get current time",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "12:00"}, nil
					},
				},
			},
			OnStepFinish: func(step stream.StepResultData) {
				stepFinishCount++
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Collect all events
		var events []stream.Event
		var toolResultEvents int
		var startStepEvents int
		for event := range result.FullStream {
			events = append(events, event)
			if event.Type == stream.EventToolResult {
				toolResultEvents++
			}
			if event.Type == stream.EventStartStep {
				startStepEvents++
			}
		}

		// Should have start-step events for each step
		if startStepEvents != 2 {
			t.Errorf("expected 2 start-step events, got %d", startStepEvents)
		}

		// Should have tool result event
		if toolResultEvents != 1 {
			t.Errorf("expected 1 tool result event, got %d", toolResultEvents)
		}

		// Final text should be from step 2
		if result.Text() != "The time is 12:00" {
			t.Errorf("expected 'The time is 12:00', got '%s'", result.Text())
		}

		// OnStepFinish should be called twice
		if stepFinishCount != 2 {
			t.Errorf("expected onStepFinish to be called 2 times, got %d", stepFinishCount)
		}

		// Check accumulated usage
		usage := result.Usage()
		if usage.InputTotal() != 25 {
			t.Errorf("expected 25 prompt tokens, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 18 {
			t.Errorf("expected 18 completion tokens, got %d", usage.OutputTotal())
		}
	})
}

func TestStopConditions(t *testing.T) {
	t.Run("StepCountIs should stop after N steps", func(t *testing.T) {
		condition := StepCountIs(3)

		// 0 steps - should not stop
		if condition(nil) {
			t.Error("should not stop at 0 steps")
		}

		// 2 steps - should not stop
		if condition([]StepResult{{}, {}}) {
			t.Error("should not stop at 2 steps")
		}

		// 3 steps - should stop
		if !condition([]StepResult{{}, {}, {}}) {
			t.Error("should stop at 3 steps")
		}
	})

	t.Run("HasToolCall should stop when tool is called", func(t *testing.T) {
		condition := HasToolCall("target_tool")

		// No steps - should not stop
		if condition(nil) {
			t.Error("should not stop with no steps")
		}

		// Step with different tool - should not stop
		steps := []StepResult{
			{Content: []ContentPart{ToolCallContentPart{ToolCall: stream.ToolCall{Name: "other_tool"}}}},
		}
		if condition(steps) {
			t.Error("should not stop with different tool")
		}

		// Step with target tool - should stop
		steps = []StepResult{
			{Content: []ContentPart{ToolCallContentPart{ToolCall: stream.ToolCall{Name: "target_tool"}}}},
		}
		if !condition(steps) {
			t.Error("should stop when target tool is called")
		}
	})

	t.Run("IsStopConditionMet should return true if any condition is met", func(t *testing.T) {
		conditions := []StopCondition{
			StepCountIs(5),
			HasToolCall("stop_tool"),
		}

		// Neither condition met
		steps := []StepResult{{}, {}}
		if IsStopConditionMet(conditions, steps) {
			t.Error("should not stop when no conditions are met")
		}

		// Step count met
		steps = []StepResult{{}, {}, {}, {}, {}}
		if !IsStopConditionMet(conditions, steps) {
			t.Error("should stop when step count is met")
		}

		// Tool call met (even though step count not met)
		steps = []StepResult{
			{Content: []ContentPart{ToolCallContentPart{ToolCall: stream.ToolCall{Name: "stop_tool"}}}},
		}
		if !IsStopConditionMet(conditions, steps) {
			t.Error("should stop when tool call condition is met")
		}
	})
}

// Tests for GenerateText with Output (structured output).
// Source: ai-sdk/packages/ai/src/generate-text/generate-text.test.ts (output option)

func TestGenerateText_Output(t *testing.T) {
	objectSchema, err := schema.FromType(struct {
		Sentiment  string  `json:"sentiment"`
		Confidence float64 `json:"confidence"`
	}{})
	if err != nil {
		t.Fatalf("schema build: %v", err)
	}

	t.Run("parses object output when finish reason is stop", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse(
				`{"sentiment":"positive","confidence":0.9}`,
				testutil.MockUsage(5, 10),
			),
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model: model,
			Output: output.Object(output.ObjectOptions{
				Schema: objectSchema,
				Name:   "Sentiment",
			}),
			Messages: []message.Message{
				message.NewUserMessage("Analyze: I love this"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		obj, ok := result.Output.(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", result.Output)
		}
		if obj["sentiment"] != "positive" {
			t.Errorf("sentiment: got %v", obj["sentiment"])
		}
		if obj["confidence"] != 0.9 {
			t.Errorf("confidence: got %v", obj["confidence"])
		}
	})

	t.Run("sends ResponseFormat on every step", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse(
					"call_1", "lookup",
					map[string]string{"q": "x"},
					testutil.MockUsage(5, 5),
				),
				testutil.MockTextResponse(`{"sentiment":"neutral","confidence":0.5}`, testutil.MockUsage(5, 10)),
			},
		})

		_, err := GenerateText(context.Background(), stream.Input{
			Model:    model,
			MaxSteps: 3,
			Output: output.Object(output.ObjectOptions{
				Schema: objectSchema,
			}),
			Tools: tool.Set{
				"lookup": tool.Tool{
					Name:        "lookup",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "neutral signal"}, nil
					},
				},
			},
			Messages: []message.Message{
				message.NewUserMessage("Decide"),
			},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(model.DoStreamCalls) != 2 {
			t.Fatalf("expected 2 stream calls, got %d", len(model.DoStreamCalls))
		}
		for i, call := range model.DoStreamCalls {
			if call.ResponseFormat == nil {
				t.Errorf("step %d: ResponseFormat was nil", i)
				continue
			}
			if call.ResponseFormat.Type != "json" {
				t.Errorf("step %d: ResponseFormat.Type = %q, want json", i, call.ResponseFormat.Type)
			}
		}
	})

	t.Run("combines tools with output - parses on final stop step", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				testutil.MockToolCallResponse(
					"call_1", "lookup",
					map[string]string{"q": "x"},
					testutil.MockUsage(5, 5),
				),
				testutil.MockTextResponse(`{"sentiment":"positive","confidence":0.8}`, testutil.MockUsage(5, 10)),
			},
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model:    model,
			MaxSteps: 3,
			Output: output.Object(output.ObjectOptions{
				Schema: objectSchema,
			}),
			Tools: tool.Set{
				"lookup": tool.Tool{
					Name:        "lookup",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: "x"}, nil
					},
				},
			},
			Messages: []message.Message{message.NewUserMessage("Decide")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Steps) != 2 {
			t.Fatalf("expected 2 steps, got %d", len(result.Steps))
		}
		if result.Steps[0].FinishReason != stream.FinishReasonToolCalls {
			t.Errorf("step 0 finish: got %s", result.Steps[0].FinishReason)
		}
		if result.Steps[1].FinishReason != stream.FinishReasonStop {
			t.Errorf("step 1 finish: got %s", result.Steps[1].FinishReason)
		}
		obj, ok := result.Output.(map[string]any)
		if !ok {
			t.Fatalf("expected parsed object, got %T (%v)", result.Output, result.Output)
		}
		if obj["sentiment"] != "positive" {
			t.Errorf("sentiment: got %v", obj["sentiment"])
		}
	})

	t.Run("does not parse output when finish reason is not stop", func(t *testing.T) {
		// MaxSteps=1 forces stop after the first step, even though it's a tool call.
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockToolCallResponse(
				"call_1", "noop",
				map[string]string{},
				testutil.MockUsage(5, 5),
			),
		})

		result, err := GenerateText(context.Background(), stream.Input{
			Model:    model,
			MaxSteps: 1,
			Output: output.Object(output.ObjectOptions{
				Schema: objectSchema,
			}),
			Tools: tool.Set{
				"noop": tool.Tool{
					Name:        "noop",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
						return tool.Result{Output: ""}, nil
					},
				},
			},
			Messages: []message.Message{message.NewUserMessage("x")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result.Output != nil {
			t.Errorf("Output should be nil when last step did not stop, got %v", result.Output)
		}
	})
}

func TestStreamText_Output(t *testing.T) {
	objectSchema, err := schema.FromType(struct {
		Greeting string `json:"greeting"`
	}{})
	if err != nil {
		t.Fatalf("schema build: %v", err)
	}

	t.Run("parses object output via Result.Output closure", func(t *testing.T) {
		model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{
			StreamResponse: testutil.MockTextResponse(
				`{"greeting":"hi"}`,
				testutil.MockUsage(2, 4),
			),
		})

		res, err := StreamText(context.Background(), stream.Input{
			Model: model,
			Output: output.Object(output.ObjectOptions{
				Schema: objectSchema,
			}),
			Messages: []message.Message{message.NewUserMessage("greet")},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain the stream
		for range res.FullStream {
		}

		obj, ok := res.Output().(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any, got %T", res.Output())
		}
		if obj["greeting"] != "hi" {
			t.Errorf("greeting: got %v", obj["greeting"])
		}
	})
}

// Regression for ai-sdk #14281: invalid/non-object JSON in a tool-call
// input should be substituted with {} when building the response message
// so subsequent steps can re-serialize cleanly.
func TestBuildStepMessages_InvalidToolInputSubstitutedWithEmptyObject(t *testing.T) {
	tests := []struct {
		name  string
		input json.RawMessage
		want  string
	}{
		{"empty", json.RawMessage(""), `{}`},
		{"malformed", json.RawMessage(`{"k":`), `{}`},
		{"non-object array", json.RawMessage(`[1,2]`), `{}`},
		{"non-object string", json.RawMessage(`"hi"`), `{}`},
		{"valid object", json.RawMessage(`{"k":"v"}`), `{"k":"v"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			msgs := buildStepMessages("", nil, []stream.ToolCall{
				{ID: "c1", Name: "t", Input: tc.input},
			}, nil)
			if len(msgs) != 1 {
				t.Fatalf("expected 1 assistant message, got %d", len(msgs))
			}
			var got string
			for _, p := range msgs[0].Content.Parts {
				if tcp, ok := p.(message.ToolCallPart); ok {
					got = string(tcp.Input)
				}
			}
			if got != tc.want {
				t.Errorf("input = %q, want %q", got, tc.want)
			}
		})
	}
}
