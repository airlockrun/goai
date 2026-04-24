package middleware

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// Tests translated from ai-sdk's add-tool-input-examples-middleware.test.ts.

func TestAddToolInputExamples_AppendsDefaults(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{}
	in := &stream.CallOptions{
		Tools: []tool.Tool{
			{
				Name:        "get_weather",
				Description: "Fetch the weather.",
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"location":"Berlin"}`)},
					{Input: json.RawMessage(`{"location":"Paris"}`)},
				},
			},
		},
	}
	out := mw.TransformOptions(in)
	if out == in {
		t.Fatal("expected a new CallOptions (copy), got the same pointer")
	}
	got := out.Tools[0]
	wantDesc := "Fetch the weather.\n\nInput Examples:\n{\"location\":\"Berlin\"}\n{\"location\":\"Paris\"}"
	if got.Description != wantDesc {
		t.Errorf("Description = %q, want %q", got.Description, wantDesc)
	}
	if len(got.InputExamples) != 0 {
		t.Errorf("InputExamples should be cleared by default, got %d", len(got.InputExamples))
	}
	// Original untouched.
	if len(in.Tools[0].InputExamples) != 2 {
		t.Errorf("original tool was mutated; InputExamples len = %d", len(in.Tools[0].InputExamples))
	}
}

func TestAddToolInputExamples_UsesCustomPrefixAndFormat(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{
		Prefix: "Examples:",
		Format: func(ex tool.ToolInputExample, idx int) string {
			return "#" + itoa(idx) + " " + string(ex.Input)
		},
	}
	out := mw.TransformOptions(&stream.CallOptions{
		Tools: []tool.Tool{
			{
				Name:        "t",
				Description: "Desc",
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"x":1}`)},
					{Input: json.RawMessage(`{"y":2}`)},
				},
			},
		},
	})
	got := out.Tools[0].Description
	want := "Desc\n\nExamples:\n#0 {\"x\":1}\n#1 {\"y\":2}"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

func TestAddToolInputExamples_KeepExamplesRetainsField(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{KeepExamples: true}
	out := mw.TransformOptions(&stream.CallOptions{
		Tools: []tool.Tool{
			{
				Name:        "t",
				Description: "d",
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"a":true}`)},
				},
			},
		},
	})
	got := out.Tools[0]
	if len(got.InputExamples) != 1 {
		t.Errorf("InputExamples len = %d, want 1 (KeepExamples=true)", len(got.InputExamples))
	}
	if !strings.Contains(got.Description, "Input Examples:") {
		t.Errorf("Description missing examples block: %q", got.Description)
	}
}

func TestAddToolInputExamples_NoExamplesPassThrough(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{}
	in := &stream.CallOptions{
		Tools: []tool.Tool{
			{Name: "t", Description: "d"},
			{Name: "u", Description: "e"},
		},
	}
	out := mw.TransformOptions(in)
	if out != in {
		t.Error("expected pointer-equal pass-through when no tool has examples")
	}
}

func TestAddToolInputExamples_SkipsProviderTools(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{}
	in := &stream.CallOptions{
		Tools: []tool.Tool{
			{
				Type:       "provider",
				ProviderID: "google.google_search",
				Name:       "google_search",
				// Should not be touched even if InputExamples is present.
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"q":"cats"}`)},
				},
				Description: "Search",
			},
		},
	}
	out := mw.TransformOptions(in)
	if out != in {
		t.Error("expected pointer-equal pass-through when all tools are provider-defined")
	}
}

func TestAddToolInputExamples_EmptyDescriptionBecomesExamplesBlock(t *testing.T) {
	mw := &AddToolInputExamplesMiddleware{}
	out := mw.TransformOptions(&stream.CallOptions{
		Tools: []tool.Tool{
			{
				Name: "t",
				InputExamples: []tool.ToolInputExample{
					{Input: json.RawMessage(`{"k":"v"}`)},
				},
			},
		},
	})
	got := out.Tools[0].Description
	want := "Input Examples:\n{\"k\":\"v\"}"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

// itoa is a tiny helper so we don't import strconv for a single call.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
