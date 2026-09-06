package goai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/testutil"
	"github.com/airlockrun/goai/tool"
)

func TestCoreEmptyStreamFailure(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "generate"
		if streaming {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				events []stream.Event
			}{
				{"empty", nil},
				{"text framing", []stream.Event{{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}, {Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{}}, {Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}}},
				{"reasoning framing", []stream.Event{{Type: stream.EventReasoningStart, Data: stream.ReasoningStartEvent{}}}},
				{"metadata", []stream.Event{{Type: stream.EventStart, Data: stream.StartEvent{}}, {Type: stream.EventStartStep, Data: stream.StartStepEvent{}}, {Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: "metadata"}}}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					attempts, ended := 0, false
					model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{DoStreamFunc: func(context.Context, *stream.CallOptions) (<-chan stream.Event, error) {
						attempts++
						ch := make(chan stream.Event, len(tc.events))
						for _, event := range tc.events {
							ch <- event
						}
						close(ch)
						return ch, nil
					}})
					input := stream.Input{Model: model, OnEnd: func(stream.OnEndData) { ended = true }}
					var err error
					if streaming {
						result, setupErr := StreamText(context.Background(), input)
						if setupErr != nil {
							t.Fatal(setupErr)
						}
						for event := range result.FullStream {
							if event.Type == stream.EventFinish {
								t.Error("unexpected successful finish")
							}
						}
						_, err = result.Text()
						if reason, reasonErr := result.FinishReason(); reason != stream.FinishReasonError || reasonErr == nil {
							t.Errorf("finish = %s, %v", reason, reasonErr)
						}
					} else {
						_, err = GenerateText(context.Background(), input)
					}
					if !errors.Is(err, goaierrors.ErrInvalidResponse) || attempts != 1 || ended {
						t.Fatalf("error = %v, attempts = %d, ended = %v", err, attempts, ended)
					}
				})
			}
		})
	}
}

func TestCoreProviderReplayOrder(t *testing.T) {
	for _, mode := range []string{"generate", "stream"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			metadata := map[string]any{"anthropic": map[string]any{"rawBlock": json.RawMessage(`{"type":"web_search_tool_result","tool_use_id":"hosted","content":[]}`)}}
			model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{DoStreamFunc: func(_ context.Context, opts *stream.CallOptions) (<-chan stream.Event, error) {
				calls++
				var events []stream.Event
				if calls == 1 {
					events = []stream.Event{
						{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "Before"}},
						{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "hosted", ToolName: "search", Input: json.RawMessage(`{}`), ProviderExecuted: true}},
						{Type: stream.EventToolResult, Data: stream.ToolResultEvent{ToolCallID: "hosted", ToolName: "search", Output: message.JSONOutput{Value: []any{}}, ProviderExecuted: true, ProviderMetadata: metadata}},
						{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: "After"}},
						{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "local", ToolName: "local", Input: json.RawMessage(`{}`)}},
						{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonToolCalls}},
					}
				} else {
					// Exercise persistence as well as the next-request conversion.
					raw, err := json.Marshal(opts.Messages)
					if err != nil {
						return nil, err
					}
					var messages []message.Message
					if err := json.Unmarshal(raw, &messages); err != nil {
						return nil, err
					}
					body, _, _, err := anthropic.BuildRequestBody(anthropic.Config{}, "claude-sonnet-4-6", &stream.CallOptions{Messages: messages})
					if err != nil {
						return nil, err
					}
					var wire struct {
						Messages []struct {
							Content []struct{ Type, Text string }
						}
					}
					if err := json.Unmarshal(body, &wire); err != nil {
						return nil, err
					}
					var types []string
					for _, part := range wire.Messages[1].Content {
						types = append(types, part.Type+":"+part.Text)
					}
					want := []string{"text:Before", "server_tool_use:", "web_search_tool_result:", "text:After", "tool_use:"}
					if !reflect.DeepEqual(types, want) {
						t.Errorf("replay order = %v, want %v", types, want)
					}
					events = []stream.Event{{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonStop}}}
				}
				ch := make(chan stream.Event, len(events))
				for _, event := range events {
					ch <- event
				}
				close(ch)
				return ch, nil
			}})
			input := stream.Input{Model: model, Messages: []message.Message{message.NewUserMessage("search")}, MaxSteps: 2, Tools: tool.Set{
				"local": Tool("local", "", json.RawMessage(`{}`), func(context.Context, json.RawMessage, tool.CallOptions) (tool.Result, error) {
					return tool.Result{Output: "done"}, nil
				}),
			}}
			if mode == "stream" {
				result, err := StreamText(context.Background(), input)
				if err != nil {
					t.Fatal(err)
				}
				for range result.FullStream {
				}
				if _, err := result.Text(); err != nil {
					t.Fatal(err)
				}
			} else if _, err := GenerateText(context.Background(), input); err != nil {
				t.Fatal(err)
			}
			if calls != 2 {
				t.Fatalf("model calls = %d", calls)
			}
		})
	}
}

func TestCoreProviderExecutedTools(t *testing.T) {
	for _, mode := range []stream.ToolCallExecutionMode{stream.ToolCallExecutionSync, stream.ToolCallExecutionAsync} {
		t.Run(string(mode), func(t *testing.T) {
			for _, streaming := range []bool{false, true} {
				name := "generate"
				if streaming {
					name = "stream"
				}
				t.Run(name, func(t *testing.T) {
					localRuns := 0
					var step stream.StepResultData
					model := testutil.NewMockLanguageModel(testutil.MockLanguageModelOptions{DoStreamFunc: func(context.Context, *stream.CallOptions) (<-chan stream.Event, error) {
						events := []stream.Event{
							{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "hosted", ToolName: "search", Input: json.RawMessage(`{}`), ProviderExecuted: true, ProviderMetadata: map[string]any{"itemId": "remote"}}},
							{Type: stream.EventToolResult, Data: stream.ToolResultEvent{ToolCallID: "hosted", ToolName: "search", Output: message.JSONOutput{Value: map[string]any{"found": true}}, ProviderExecuted: true, ProviderMetadata: map[string]any{"itemId": "result"}}},
							{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: "local", ToolName: "local", Input: json.RawMessage(`{}`)}},
							{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: stream.FinishReasonToolCalls, Usage: stream.Usage{InputTokens: stream.InputTokens{Total: stream.IntPtr(-1)}, Raw: map[string]any{"tokens": -1}}}},
						}
						ch := make(chan stream.Event, len(events))
						for _, e := range events {
							ch <- e
						}
						close(ch)
						return ch, nil
					}})
					input := stream.Input{Model: model, ToolCallExecutionMode: mode, Tools: tool.Set{
						"search": Tool("search", "", json.RawMessage(`{}`), func(context.Context, json.RawMessage, tool.CallOptions) (tool.Result, error) {
							t.Error("hosted tool executed locally")
							return tool.Result{}, nil
						}),
						"local": Tool("local", "", json.RawMessage(`{}`), func(context.Context, json.RawMessage, tool.CallOptions) (tool.Result, error) {
							localRuns++
							return tool.Result{Output: "done"}, nil
						}),
					}, RefineToolInput: func(name string, input json.RawMessage) (json.RawMessage, error) {
						if name == "search" {
							t.Error("refined hosted tool")
						}
						return input, nil
					}, OnStepEnd: func(s stream.StepResultData) { step = s }}
					var usage stream.Usage
					if streaming {
						result, err := StreamText(context.Background(), input)
						if err != nil {
							t.Fatal(err)
						}
						hostedResults := 0
						for event := range result.FullStream {
							if e, ok := event.Data.(stream.ToolResultEvent); ok && e.ProviderExecuted {
								hostedResults++
							}
						}
						if hostedResults != 1 {
							t.Errorf("hosted results emitted %d times", hostedResults)
						}
						if _, err := result.Text(); err != nil {
							t.Fatal(err)
						}
						usage, _ = result.Usage()
					} else {
						result, err := GenerateText(context.Background(), input)
						if err != nil {
							t.Fatal(err)
						}
						usage = result.Usage
					}
					if localRuns != 1 || usage.InputTotal() != 0 || usage.Raw != nil {
						t.Fatalf("local runs = %d, usage = %+v", localRuns, usage)
					}
					s := step.(*StepResult)
					if s.Usage.Raw["tokens"] != -1 || s.Usage.InputTotal() != 0 || len(s.ToolResults()) != 2 {
						t.Fatalf("step = %+v", s)
					}
					messages := s.Response.Messages
					encoded, err := json.Marshal(messages)
					if err != nil {
						t.Fatal(err)
					}
					var replay []message.Message
					if err := json.Unmarshal(encoded, &replay); err != nil {
						t.Fatal(err)
					}
					if len(replay) != 2 || replay[0].Role != message.RoleAssistant || replay[1].Role != message.RoleTool {
						t.Fatalf("replay = %s", encoded)
					}
					call := replay[0].Content.Parts[0].(message.ToolCallPart)
					result := replay[0].Content.Parts[1].(message.ToolResultPart)
					if !call.ProviderExecuted || call.ProviderOptions["itemId"] != "remote" || !result.ProviderExecuted || result.ProviderOptions["itemId"] != "result" {
						t.Fatalf("markers lost: %s", encoded)
					}
				})
			}
		})
	}
}
