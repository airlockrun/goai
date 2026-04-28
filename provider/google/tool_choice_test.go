package google

import (
	"reflect"
	"testing"
)

// Translated from ai-sdk/packages/google/src/google-prepare-tools.test.ts
// (the toolChoice cases). Gemini uses toolConfig.functionCallingConfig.mode
// instead of a top-level tool_choice field.

func TestConvertToolChoice(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want *geminiToolConfig
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty string", in: "", want: nil},

		{name: "string auto → AUTO", in: "auto", want: modeOnly("AUTO")},
		{name: "string required → ANY", in: "required", want: modeOnly("ANY")},
		{name: "string none → NONE", in: "none", want: modeOnly("NONE")},

		{name: "object auto", in: map[string]any{"type": "auto"}, want: modeOnly("AUTO")},
		{name: "object required → ANY", in: map[string]any{"type": "required"}, want: modeOnly("ANY")},
		{name: "object none", in: map[string]any{"type": "none"}, want: modeOnly("NONE")},

		{
			name: "object tool with toolName → ANY + allowedFunctionNames",
			in:   map[string]any{"type": "tool", "toolName": "calculator"},
			want: &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode:                 "ANY",
					AllowedFunctionNames: []string{"calculator"},
				},
			},
		},
		{
			name: "bare tool name → ANY + allowedFunctionNames",
			in:   "calculator",
			want: &geminiToolConfig{
				FunctionCallingConfig: &geminiFunctionCallingConfig{
					Mode:                 "ANY",
					AllowedFunctionNames: []string{"calculator"},
				},
			},
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
