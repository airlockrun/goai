package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/openai/src/chat/openai-chat-prepare-tools.test.ts
// and openai-responses-prepare-tools.test.ts (the toolChoice cases).

func TestConvertChatToolChoice(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty string", in: "", want: nil},

		// Bare-string passthrough.
		{name: "string auto", in: "auto", want: "auto"},
		{name: "string none", in: "none", want: "none"},
		{name: "string required", in: "required", want: "required"},

		// ai-sdk shape {type: 'auto'} → bare string.
		{name: "object auto unwraps", in: map[string]any{"type": "auto"}, want: "auto"},
		{name: "object none unwraps", in: map[string]any{"type": "none"}, want: "none"},
		{name: "object required unwraps", in: map[string]any{"type": "required"}, want: "required"},

		// Specific tool: ai-sdk shape and bare-name → wire {type: function, function: {name}}.
		{
			name: "object tool with toolName → function",
			in:   map[string]any{"type": "tool", "toolName": "calculator"},
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},
		{
			name: "bare tool name → function",
			in:   "calculator",
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},

		// Wire-form passthrough.
		{
			name: "wire {type: function, function: {name}}",
			in:   map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := convertChatToolChoice(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}

// End-to-end: ToolChoice on the wire payload for chat completions.
func TestChatModel_ToolChoiceWireForm(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{name: "string auto → bare string", in: "auto", want: "auto"},
		{name: "string required → bare string", in: "required", want: "required"},
		{name: "string none → bare string", in: "none", want: "none"},
		{
			name: "ai-sdk {type: tool, toolName} → wire function",
			in:   map[string]any{"type": "tool", "toolName": "calculator"},
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},
		{
			name: "bare tool name → wire function",
			in:   "calculator",
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := captureChatRequestBody(t, &stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("hi")},
				Tools: []tool.Tool{{
					Name:        "calculator",
					Description: "calc",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				}},
				ToolChoice: tc.in,
			})
			got, has := body["tool_choice"]
			if !has {
				t.Fatalf("expected tool_choice on wire, got none. body=%v", body)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("tool_choice: got %#v, want %#v", got, tc.want)
			}
		})
	}
}

func captureChatRequestBody(t *testing.T, callOpts *stream.CallOptions) map[string]any {
	t.Helper()
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Minimal stream so the model finishes cleanly.
		chunks := []string{
			`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
			`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		}
		for _, c := range chunks {
			w.Write([]byte("data: " + c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(provider.Options{APIKey: "test", BaseURL: server.URL})
	model := p.Chat("gpt-4o-mini")
	events, err := model.Stream(context.Background(), callOpts)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	return captured
}

// Translated from ai-sdk/packages/openai/src/responses/openai-responses-prepare-tools.test.ts
// (the toolChoice cases).

func TestConvertResponsesToolChoice(t *testing.T) {
	calc := tool.Tool{
		Name:        "calculator",
		Description: "calc",
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}
	webSearch := WebSearch()
	custom := Custom(CustomOptions{Name: "regex_grammar"})
	toolSearch := ToolSearch(ToolSearchOptions{Execution: "auto"})
	tools := []tool.Tool{calc, webSearch, custom, toolSearch}

	cases := []struct {
		name string
		in   any
		want any
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty string", in: "", want: nil},

		// Bare-string passthrough (OpenAI Responses accepts these natively).
		{name: "string auto", in: "auto", want: "auto"},
		{name: "string none", in: "none", want: "none"},
		{name: "string required", in: "required", want: "required"},

		// ai-sdk discriminated-union shapes → bare string.
		{name: "object auto unwraps", in: map[string]any{"type": "auto"}, want: "auto"},
		{name: "object none unwraps", in: map[string]any{"type": "none"}, want: "none"},
		{name: "object required unwraps", in: map[string]any{"type": "required"}, want: "required"},

		// Function tool by name (ai-sdk shape and bare).
		{name: "object tool with toolName → function", in: map[string]any{"type": "tool", "toolName": "calculator"}, want: map[string]any{"type": "function", "name": "calculator"}},
		{name: "bare tool name → function", in: "calculator", want: map[string]any{"type": "function", "name": "calculator"}},
		{name: "object tool unknown name → function", in: map[string]any{"type": "tool", "toolName": "unknown"}, want: map[string]any{"type": "function", "name": "unknown"}},

		// Hosted tools by name → wire-format type field.
		{name: "tool name web_search → hosted", in: map[string]any{"type": "tool", "toolName": "web_search"}, want: map[string]any{"type": "web_search"}},
		{name: "tool name openai.web_search (providerID) → hosted", in: map[string]any{"type": "tool", "toolName": "openai.web_search"}, want: map[string]any{"type": "web_search"}},
		{name: "tool name tool_search → hosted", in: map[string]any{"type": "tool", "toolName": "tool_search"}, want: map[string]any{"type": "tool_search"}},

		// Custom tool routes to {type: "custom", name: X}.
		{name: "tool name regex_grammar (custom) → custom", in: map[string]any{"type": "tool", "toolName": "regex_grammar"}, want: map[string]any{"type": "custom", "name": "regex_grammar"}},

		// Wire-form passthrough (callers that already emit the wire shape).
		{name: "wire {type: function, name}", in: map[string]any{"type": "function", "name": "calculator"}, want: map[string]any{"type": "function", "name": "calculator"}},
		{name: "wire {type: web_search}", in: map[string]any{"type": "web_search"}, want: map[string]any{"type": "web_search"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := convertResponsesToolChoice(tc.in, tools)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}
