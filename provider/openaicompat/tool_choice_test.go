package openaicompat

import (
	"reflect"
	"testing"
)

// Translated from ai-sdk/packages/openai/src/chat/openai-chat-prepare-tools.test.ts
// (the toolChoice cases). Every OpenAI-compatible provider that builds on this
// package inherits the translation.

func TestConvertToolChoice(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty string", in: "", want: nil},

		{name: "string auto", in: "auto", want: "auto"},
		{name: "string none", in: "none", want: "none"},
		{name: "string required", in: "required", want: "required"},

		{name: "object auto unwraps", in: map[string]any{"type": "auto"}, want: "auto"},
		{name: "object none unwraps", in: map[string]any{"type": "none"}, want: "none"},
		{name: "object required unwraps", in: map[string]any{"type": "required"}, want: "required"},

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
		{
			name: "wire {type: function, function: {name}} passthrough",
			in:   map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
			want: map[string]any{"type": "function", "function": map[string]any{"name": "calculator"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := convertToolChoice(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}
