package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Translated from ai-sdk/packages/anthropic/src/anthropic-prepare-tools.test.ts
// (the toolChoice cases). Anthropic's API rejects bare strings on tool_choice
// — the provider must translate them. ai-sdk's TS prompt layer does this via a
// typed discriminated union; goai keeps stream.CallOptions.ToolChoice as
// `any`, so the translation lives in the provider.

func TestConvertAnthropicToolChoice(t *testing.T) {
	cases := []struct {
		name      string
		in        any
		want      any
		dropTools bool
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty string", in: "", want: nil},

		// goai loose-string forms (matches stream.CallOptions.ToolChoice doc).
		{name: "string auto", in: "auto", want: map[string]any{"type": "auto"}},
		{name: "string required", in: "required", want: map[string]any{"type": "any"}},
		{name: "string none", in: "none", want: nil, dropTools: true},
		{name: "string specific tool name", in: "calculator", want: map[string]any{"type": "tool", "name": "calculator"}},

		// ai-sdk discriminated-union shapes (LanguageModelV4ToolChoice).
		{name: "object auto", in: map[string]any{"type": "auto"}, want: map[string]any{"type": "auto"}},
		{name: "object required → any", in: map[string]any{"type": "required"}, want: map[string]any{"type": "any"}},
		{name: "object none", in: map[string]any{"type": "none"}, want: nil, dropTools: true},
		{
			name: "object tool with toolName (ai-sdk shape)",
			in:   map[string]any{"type": "tool", "toolName": "search"},
			want: map[string]any{"type": "tool", "name": "search"},
		},

		// Anthropic wire-form passthrough (existing callers — see chat_test.go
		// "forceJSONToolChoice overrides caller ToolChoice" using {"type":"any"}).
		{name: "wire-form any passthrough", in: map[string]any{"type": "any"}, want: map[string]any{"type": "any"}},
		{
			name: "wire-form tool with name",
			in:   map[string]any{"type": "tool", "name": "search"},
			want: map[string]any{"type": "tool", "name": "search"},
		},

		// map[string]string variant (used in existing tests).
		{name: "map[string]string any", in: map[string]string{"type": "any"}, want: map[string]any{"type": "any"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, drop := convertAnthropicToolChoice(tc.in)
			if drop != tc.dropTools {
				t.Errorf("dropTools: got %v, want %v", drop, tc.dropTools)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("choice: got %#v, want %#v", got, tc.want)
			}
		})
	}
}

// Reproduces the bug reported by sol: passing ToolChoice: "auto" to the
// Anthropic provider previously hit Anthropic's
//
//	tool_choice: Input should be a valid dictionary or object to extract fields from
//
// because the provider forwarded the bare string. Verifies the wire payload
// now contains the object form.
func TestBuildRequest_ToolChoiceWireForm(t *testing.T) {
	cases := []struct {
		name           string
		toolChoice     any
		wantToolChoice map[string]any
		wantToolsLen   int
	}{
		{
			name:           "string auto → object auto",
			toolChoice:     "auto",
			wantToolChoice: map[string]any{"type": "auto"},
			wantToolsLen:   1,
		},
		{
			name:           "string required → object any",
			toolChoice:     "required",
			wantToolChoice: map[string]any{"type": "any"},
			wantToolsLen:   1,
		},
		{
			name:           "string tool name → object tool",
			toolChoice:     "calculator",
			wantToolChoice: map[string]any{"type": "tool", "name": "calculator"},
			wantToolsLen:   1,
		},
		{
			name:           "ai-sdk shape {type: tool, toolName: ...} → wire {type: tool, name: ...}",
			toolChoice:     map[string]any{"type": "tool", "toolName": "calculator"},
			wantToolChoice: map[string]any{"type": "tool", "name": "calculator"},
			wantToolsLen:   1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := captureRequestBody(t, &stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("hi")},
				Tools: []tool.Tool{{
					Name:        "calculator",
					Description: "calc",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				}},
				ToolChoice: tc.toolChoice,
			})

			tools, _ := body["tools"].([]any)
			if len(tools) != tc.wantToolsLen {
				t.Errorf("tools length: got %d, want %d", len(tools), tc.wantToolsLen)
			}

			gotChoice, ok := body["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("expected tool_choice to be object, got %T (%v)", body["tool_choice"], body["tool_choice"])
			}
			if !reflect.DeepEqual(gotChoice, tc.wantToolChoice) {
				t.Errorf("tool_choice: got %#v, want %#v", gotChoice, tc.wantToolChoice)
			}
		})
	}
}

// "none" is special: Anthropic has no native "none" tool_choice. ai-sdk
// models it by dropping both tools[] and tool_choice. Verify goai matches.
func TestBuildRequest_ToolChoiceNoneDropsTools(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   any
	}{
		{name: "string none", in: "none"},
		{name: "object none", in: map[string]any{"type": "none"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := captureRequestBody(t, &stream.CallOptions{
				Messages: []message.Message{message.NewUserMessage("hi")},
				Tools: []tool.Tool{{
					Name:        "calculator",
					Description: "calc",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				}},
				ToolChoice: tc.in,
			})

			if _, has := body["tools"]; has {
				t.Errorf("expected tools[] to be dropped, got %v", body["tools"])
			}
			if _, has := body["tool_choice"]; has {
				t.Errorf("expected tool_choice to be omitted, got %v", body["tool_choice"])
			}
		})
	}
}

// captureRequestBody runs a single streaming call against a stub server and
// returns the parsed JSON request body. Mirrors chat_test.go's runCapture.
func captureRequestBody(t *testing.T, callOpts *stream.CallOptions) map[string]any {
	t.Helper()
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Minimal complete stream so the model finishes cleanly.
		chunks := []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`,
			`{"type":"message_stop"}`,
		}
		for _, c := range chunks {
			w.Write([]byte("data: " + c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	provider := createTestProvider(server.URL)
	model := provider.Model("claude-3-haiku-20240307")
	events, err := model.Stream(context.Background(), callOpts)
	if err != nil {
		t.Fatal(err)
	}
	for range events {
	}
	return captured
}
