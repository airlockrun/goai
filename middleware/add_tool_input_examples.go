package middleware

import (
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// AddToolInputExamplesMiddleware serializes each function tool's
// InputExamples into its Description, for providers that don't natively
// accept an input_examples field. Mirrors ai-sdk's
// addToolInputExamplesMiddleware.
//
// Provider-defined tools (Type == "provider") are skipped — examples on
// built-in tools aren't meaningful since the provider runs them itself.
type AddToolInputExamplesMiddleware struct {
	BaseMiddleware

	// Prefix is prepended before the serialized examples list. Defaults to
	// "Input Examples:" when empty.
	Prefix string

	// Format renders a single example. Defaults to json.Marshal(ex.Input)
	// when nil.
	Format func(ex tool.ToolInputExample, index int) string

	// Remove clears the InputExamples field on the transformed tool after
	// folding them into the description. Defaults to true (matching ai-sdk).
	// Use a pointer? We stick with the zero-value semantics of ai-sdk: if
	// you want Remove=false you must opt in. ai-sdk's default is `true`, so
	// we invert via KeepExamples for Go-idiomatic zero-value defaults.
	KeepExamples bool
}

// TransformOptions copies options.Tools, appending examples to descriptions.
func (m *AddToolInputExamplesMiddleware) TransformOptions(options *stream.CallOptions) *stream.CallOptions {
	if options == nil || len(options.Tools) == 0 {
		return options
	}
	prefix := m.Prefix
	if prefix == "" {
		prefix = "Input Examples:"
	}
	format := m.Format
	if format == nil {
		format = defaultFormatExample
	}

	// Only rewrite tools that actually have examples — avoid allocating a
	// new slice when nothing would change.
	needsRewrite := false
	for _, t := range options.Tools {
		if t.Type == "" && len(t.InputExamples) > 0 {
			needsRewrite = true
			break
		}
	}
	if !needsRewrite {
		return options
	}

	newTools := make([]tool.Tool, len(options.Tools))
	for i, t := range options.Tools {
		if t.Type != "" || len(t.InputExamples) == 0 {
			newTools[i] = t
			continue
		}
		lines := make([]string, len(t.InputExamples))
		for j, ex := range t.InputExamples {
			lines[j] = format(ex, j)
		}
		examplesBlock := prefix + "\n" + strings.Join(lines, "\n")

		copied := t
		if copied.Description == "" {
			copied.Description = examplesBlock
		} else {
			copied.Description = copied.Description + "\n\n" + examplesBlock
		}
		if !m.KeepExamples {
			copied.InputExamples = nil
		}
		newTools[i] = copied
	}

	result := *options
	result.Tools = newTools
	return &result
}

func defaultFormatExample(ex tool.ToolInputExample, _ int) string {
	if len(ex.Input) == 0 {
		return "{}"
	}
	// Re-marshal to normalize whitespace (ai-sdk does JSON.stringify).
	var v any
	if err := json.Unmarshal(ex.Input, &v); err != nil {
		return string(ex.Input)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return string(ex.Input)
	}
	return string(b)
}
