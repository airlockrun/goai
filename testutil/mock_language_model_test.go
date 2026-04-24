package testutil

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/airlockrun/goai/stream"
)

// Tests for MockLanguageModel
// Source: ai-sdk/packages/ai/src/test/mock-language-model-v3.ts

func TestMockLanguageModel_Defaults(t *testing.T) {
	t.Run("should use default ID and provider", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{})

		if model.ID() != "mock-model-id" {
			t.Errorf("expected ID 'mock-model-id', got '%s'", model.ID())
		}
		if model.Provider() != "mock-provider" {
			t.Errorf("expected provider 'mock-provider', got '%s'", model.Provider())
		}
	})

	t.Run("should use custom ID and provider", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{
			ID:       "custom-model",
			Provider: "custom-provider",
		})

		if model.ID() != "custom-model" {
			t.Errorf("expected ID 'custom-model', got '%s'", model.ID())
		}
		if model.Provider() != "custom-provider" {
			t.Errorf("expected provider 'custom-provider', got '%s'", model.Provider())
		}
	})
}

func TestMockLanguageModel_Stream(t *testing.T) {
	t.Run("should return StreamResponse events", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{
			StreamResponse: MockTextResponse("Hello, world!", MockUsage(10, 20)),
		})

		events, err := model.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var collectedEvents []stream.Event
		for event := range events {
			collectedEvents = append(collectedEvents, event)
		}

		if len(collectedEvents) != 4 {
			t.Fatalf("expected 4 events, got %d", len(collectedEvents))
		}

		// Check text delta
		if textDelta, ok := collectedEvents[1].Data.(stream.TextDeltaEvent); ok {
			if textDelta.Text != "Hello, world!" {
				t.Errorf("expected text 'Hello, world!', got '%s'", textDelta.Text)
			}
		} else {
			t.Errorf("expected TextDeltaEvent, got %T", collectedEvents[1].Data)
		}
	})

	t.Run("should record DoStreamCalls", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{
			StreamResponse: MockTextResponse("test", MockUsage(1, 1)),
		})

		options := &stream.CallOptions{
			Temperature: ptrFloat64(0.7),
		}

		_, err := model.Stream(context.Background(), options)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(model.DoStreamCalls) != 1 {
			t.Fatalf("expected 1 call, got %d", len(model.DoStreamCalls))
		}
		if model.DoStreamCalls[0] != options {
			t.Error("expected recorded options to match")
		}
	})

	t.Run("should use DoStreamFunc when provided", func(t *testing.T) {
		funcCalled := false
		model := NewMockLanguageModel(MockLanguageModelOptions{
			DoStreamFunc: func(ctx context.Context, options *stream.CallOptions) (<-chan stream.Event, error) {
				funcCalled = true
				events := make(chan stream.Event, 1)
				close(events)
				return events, nil
			},
		})

		_, err := model.Stream(context.Background(), &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !funcCalled {
			t.Error("expected DoStreamFunc to be called")
		}
	})

	t.Run("should return sequential responses from StreamResponses", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{
			StreamResponses: [][]stream.Event{
				MockTextResponse("First", MockUsage(1, 1)),
				MockTextResponse("Second", MockUsage(2, 2)),
				MockTextResponse("Third", MockUsage(3, 3)),
			},
		})

		// First call
		events1, _ := model.Stream(context.Background(), &stream.CallOptions{})
		text1 := collectText(events1)
		if text1 != "First" {
			t.Errorf("expected 'First', got '%s'", text1)
		}

		// Second call
		events2, _ := model.Stream(context.Background(), &stream.CallOptions{})
		text2 := collectText(events2)
		if text2 != "Second" {
			t.Errorf("expected 'Second', got '%s'", text2)
		}

		// Third call
		events3, _ := model.Stream(context.Background(), &stream.CallOptions{})
		text3 := collectText(events3)
		if text3 != "Third" {
			t.Errorf("expected 'Third', got '%s'", text3)
		}

		// Fourth call should repeat last
		events4, _ := model.Stream(context.Background(), &stream.CallOptions{})
		text4 := collectText(events4)
		if text4 != "Third" {
			t.Errorf("expected 'Third' (repeated), got '%s'", text4)
		}
	})

	t.Run("should respect context cancellation", func(t *testing.T) {
		model := NewMockLanguageModel(MockLanguageModelOptions{
			StreamResponse: MockStreamedTextResponse(
				[]string{"chunk1", "chunk2", "chunk3", "chunk4", "chunk5"},
				MockUsage(10, 50),
			),
		})

		ctx, cancel := context.WithCancel(context.Background())
		events, err := model.Stream(ctx, &stream.CallOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Read first event
		<-events

		// Cancel context
		cancel()

		// Events channel should close
		count := 0
		for range events {
			count++
			if count > 10 {
				t.Fatal("events channel not closing after context cancellation")
			}
		}
	})
}

func TestMockTextResponse(t *testing.T) {
	events := MockTextResponse("Hello!", MockUsage(5, 10))

	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	if events[0].Type != stream.EventTextStart {
		t.Errorf("expected TextStart, got %s", events[0].Type)
	}
	if events[1].Type != stream.EventTextDelta {
		t.Errorf("expected TextDelta, got %s", events[1].Type)
	}
	if events[2].Type != stream.EventTextEnd {
		t.Errorf("expected TextEnd, got %s", events[2].Type)
	}
	if events[3].Type != stream.EventFinish {
		t.Errorf("expected Finish, got %s", events[3].Type)
	}

	finish := events[3].Data.(stream.FinishEvent)
	if finish.FinishReason != stream.FinishReasonStop {
		t.Errorf("expected FinishReasonStop, got %s", finish.FinishReason)
	}
	if finish.Usage.InputTotal() != 5 {
		t.Errorf("expected 5 prompt tokens, got %d", finish.Usage.InputTotal())
	}
}

func TestMockToolCallResponse(t *testing.T) {
	events := MockToolCallResponse("call_123", "get_weather", map[string]string{"location": "NYC"}, MockUsage(10, 5))

	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}

	if events[0].Type != stream.EventToolCall {
		t.Errorf("expected ToolCall, got %s", events[0].Type)
	}

	toolCall := events[0].Data.(stream.ToolCallEvent)
	if toolCall.ToolCallID != "call_123" {
		t.Errorf("expected call_123, got %s", toolCall.ToolCallID)
	}
	if toolCall.ToolName != "get_weather" {
		t.Errorf("expected get_weather, got %s", toolCall.ToolName)
	}

	finish := events[1].Data.(stream.FinishEvent)
	if finish.FinishReason != stream.FinishReasonToolCalls {
		t.Errorf("expected FinishReasonToolCalls, got %s", finish.FinishReason)
	}
}

func TestMockErrorResponse(t *testing.T) {
	testErr := errors.New("test error")
	events := MockErrorResponse(testErr)

	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}

	if events[0].Type != stream.EventError {
		t.Errorf("expected Error, got %s", events[0].Type)
	}

	errorEvent := events[0].Data.(stream.ErrorEvent)
	if errorEvent.Error != testErr {
		t.Error("expected same error")
	}
}

func TestMockUsage(t *testing.T) {
	usage := MockUsage(100, 50)

	if usage.InputTotal() != 100 {
		t.Errorf("expected 100 prompt tokens, got %d", usage.InputTotal())
	}
	if usage.OutputTotal() != 50 {
		t.Errorf("expected 50 completion tokens, got %d", usage.OutputTotal())
	}
	if usage.GrandTotal() != 150 {
		t.Errorf("expected 150 total tokens, got %d", usage.GrandTotal())
	}
}

// Helper functions

func collectText(events <-chan stream.Event) string {
	var b strings.Builder
	for event := range events {
		if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
			b.WriteString(delta.Text)
		}
	}
	return b.String()
}

func ptrFloat64(v float64) *float64 {
	return &v
}
