package tool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/airlockrun/goai/schema"
)

// TypedFunc is a type-safe tool implementation: it receives the decoded input
// and returns a value to be JSON-encoded as the tool result.
type TypedFunc[In, Out any] func(ctx context.Context, in In) (Out, error)

// TypedDef builds a Tool from Go types: the input schema is reflected from In,
// the output schema from Out, and the typed function is wrapped into an
// ExecuteFunc (decode In, call, encode Out). The resulting tool.Tool is the same
// value goai's generation calls accept and agentsdk registers — define a tool
// once, use it everywhere.
type TypedDef[In, Out any] struct {
	name        string
	description string
	examples    []ToolInputExample
	fn          TypedFunc[In, Out]
}

// Typed starts a type-safe tool definition. Build() reflects In→input schema and
// Out→output schema and wraps the typed Execute.
func Typed[In, Out any](name string) *TypedDef[In, Out] {
	return &TypedDef[In, Out]{name: name}
}

// Description sets the tool description.
func (d *TypedDef[In, Out]) Description(desc string) *TypedDef[In, Out] {
	d.description = desc
	return d
}

// Execute sets the type-safe implementation.
func (d *TypedDef[In, Out]) Execute(fn TypedFunc[In, Out]) *TypedDef[In, Out] {
	d.fn = fn
	return d
}

// InputExample appends an example input (the typed value is JSON-encoded).
func (d *TypedDef[In, Out]) InputExample(in In) *TypedDef[In, Out] {
	raw, err := json.Marshal(in)
	if err != nil {
		panic(fmt.Sprintf("tool.Typed(%q): InputExample: %v", d.name, err))
	}
	d.examples = append(d.examples, ToolInputExample{Input: raw})
	return d
}

// Build reflects the schemas and returns the Tool.
func (d *TypedDef[In, Out]) Build() Tool {
	name := d.name
	fn := d.fn

	def := New(name).Description(d.description).InputExamples(d.examples)

	// A tool's input is always an object of named arguments. A no-argument In
	// reflects to a non-object schema (e.g. {"type":"null"}), which strict tool /
	// MCP validators reject — use the reflected schema only when it's object-
	// typed; otherwise let Build() apply the empty-object default. Output schemas
	// may be any JSON type (a void tool legitimately has null output).
	if in := schema.MustFromType(*new(In)); in.Type == "object" {
		def = def.Schema(in.MustJSON())
	}
	def = def.OutputSchemaFromStruct(*new(Out))

	def = def.Execute(func(ctx context.Context, raw json.RawMessage, _ CallOptions) (Result, error) {
		var in In
		if len(raw) > 0 && string(raw) != "null" {
			if err := json.Unmarshal(raw, &in); err != nil {
				return Result{}, fmt.Errorf("%s: decode input: %w", name, err)
			}
		}
		if fn == nil {
			return Result{}, fmt.Errorf("%s: no execute function", name)
		}
		out, err := fn(ctx, in)
		if err != nil {
			return Result{}, fmt.Errorf("%s: %w", name, err)
		}
		buf, err := json.Marshal(out)
		if err != nil {
			return Result{}, fmt.Errorf("%s: encode output: %w", name, err)
		}
		return Result{Output: string(buf)}, nil
	})
	return def.Build()
}
