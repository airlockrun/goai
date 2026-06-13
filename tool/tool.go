// Package tool provides tool definition and execution for AI models.
// This mirrors the ai-sdk tool() and dynamicTool() functionality.
package tool

import (
	"context"
	"encoding/json"
	"sort"

	"github.com/airlockrun/goai/schema"
)

// Tool represents a tool that can be called by the AI model.
//
// Two shapes are supported:
//   - Function tool (default, Type=""): user-defined, has InputSchema +
//     optional Execute. Provider serializes as a function declaration.
//   - Provider-defined tool (Type="provider"): a built-in the provider
//     runs server-side (web_search, googleSearch, code_interpreter, ...).
//     ProviderID carries the canonical id (e.g. "google.google_search")
//     and Args carries the JSON-encoded arguments.
//
// Mirrors ai-sdk's LanguageModelV4FunctionTool | LanguageModelV4ProviderDefinedTool union.
type Tool struct {
	// Type is "" (function tool) or "provider" (provider-defined tool).
	Type string `json:"type,omitempty"`

	// ProviderID is the canonical id for provider-defined tools, e.g.
	// "google.google_search", "openai.web_search". Empty for function tools.
	ProviderID string `json:"providerID,omitempty"`

	// Args is the JSON-encoded argument payload for provider-defined tools.
	// Empty when the tool takes no args.
	Args json.RawMessage `json:"args,omitempty"`

	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"` // JSON Schema (function tools)
	// InputExamples is an optional list of example inputs that show the
	// language model what the tool input should look like. Mirrors ai-sdk's
	// LanguageModelV4FunctionTool.inputExamples. Providers that natively
	// accept examples (e.g. Anthropic via input_examples) pass them through;
	// others can use middleware.AddToolInputExamples to serialize them into
	// the description instead.
	InputExamples   []ToolInputExample `json:"inputExamples,omitempty"`
	Execute         ExecuteFunc        `json:"-"`
	ProviderOptions map[string]any     `json:"providerOptions,omitempty"`
}

// Kind classifies a Tool at the definition level. goai distinguishes the two
// definition kinds it sends to providers; the dynamic / provider-executed
// distinctions ai-sdk draws live on the tool *call* (see RawToolCall in
// parse.go), not on the definition.
type Kind string

const (
	// KindFunction is a user-defined function tool (Type==""): InputSchema +
	// optional Execute, serialized as a function declaration.
	KindFunction Kind = "function"
	// KindProviderDefined is a built-in the provider runs server-side
	// (Type=="provider"): ProviderID + Args, no Execute.
	KindProviderDefined Kind = "provider"
)

// Kind returns the tool's definition kind.
func (t Tool) Kind() Kind {
	if t.Type == string(KindProviderDefined) {
		return KindProviderDefined
	}
	return KindFunction
}

// IsProviderTool reports whether the tool is provider-defined (run server-side
// by the provider rather than via a local Execute function).
func (t Tool) IsProviderTool() bool { return t.Kind() == KindProviderDefined }

// ToolInputExample is a single example input for a tool. Mirrors ai-sdk's
// { input: JSONObject } shape.
type ToolInputExample struct {
	Input json.RawMessage `json:"input"`
}

// ExecuteFunc is the function signature for tool execution.
// It receives the parsed input and returns a result or error.
type ExecuteFunc func(ctx context.Context, input json.RawMessage, opts CallOptions) (Result, error)

// CallOptions contains options passed to tool execution.
type CallOptions struct {
	ToolCallID  string
	AbortSignal context.Context
}

// Result represents the output of a tool execution.
type Result struct {
	Output      string         `json:"output"`
	Title       string         `json:"title,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Attachments []Attachment   `json:"attachments,omitempty"`
}

// Attachment represents a file attachment in a tool result.
type Attachment struct {
	Data     string `json:"data"`     // base64 encoded
	MimeType string `json:"mimeType"` // e.g., "image/png"
	Filename string `json:"filename,omitempty"`
}

// Definition is a builder for creating tools.
type Definition struct {
	name        string
	description string
	schema      json.RawMessage
	examples    []ToolInputExample
	execute     ExecuteFunc
}

// New creates a new tool definition builder.
func New(name string) *Definition {
	return &Definition{name: name}
}

// Description sets the tool description.
func (d *Definition) Description(desc string) *Definition {
	d.description = desc
	return d
}

// Schema sets the JSON schema for the tool input.
func (d *Definition) Schema(schema json.RawMessage) *Definition {
	d.schema = schema
	return d
}

// SchemaFromStruct generates a JSON schema from a struct type.
func (d *Definition) SchemaFromStruct(v any) *Definition {
	s := schema.MustFromType(v)
	d.schema = s.MustJSON()
	return d
}

// Execute sets the execution function.
func (d *Definition) Execute(fn ExecuteFunc) *Definition {
	d.execute = fn
	return d
}

// InputExample appends a single example input to the tool definition.
// Each input should be a JSON-encoded object matching the tool's input schema.
func (d *Definition) InputExample(input json.RawMessage) *Definition {
	d.examples = append(d.examples, ToolInputExample{Input: input})
	return d
}

// InputExamples replaces the tool's example list.
func (d *Definition) InputExamples(examples []ToolInputExample) *Definition {
	d.examples = examples
	return d
}

// defaultInputSchema is sent for a function tool declared without an explicit
// input schema. The "type":"object" is required by OpenAI-compatible providers
// that reject parameter schemas lacking it.
var defaultInputSchema = json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)

// Build creates the final Tool.
func (d *Definition) Build() Tool {
	inputSchema := d.schema
	if len(inputSchema) == 0 {
		inputSchema = defaultInputSchema
	}
	return Tool{
		Name:          d.name,
		Description:   d.description,
		InputSchema:   inputSchema,
		InputExamples: d.examples,
		Execute:       d.execute,
	}
}

// Set is a collection of tools indexed by name.
type Set map[string]Tool

// Add adds a tool to the set.
func (s Set) Add(t Tool) {
	s[t.Name] = t
}

// Names returns the names of all tools in the set.
func (s Set) Names() []string {
	names := make([]string, 0, len(s))
	for name := range s {
		names = append(names, name)
	}
	return names
}

// Ordered returns tools as a slice in deterministic order.
// If activeTools is provided, returns only those tools in the specified order.
// Otherwise, returns all tools sorted alphabetically by name.
// This matches ai-sdk's behavior where tools are converted to an ordered array
// at the core level before being passed to providers.
func (s Set) Ordered(activeTools []string) []Tool {
	if len(activeTools) > 0 {
		// Use activeTools order - only include tools that exist in the set
		result := make([]Tool, 0, len(activeTools))
		for _, name := range activeTools {
			if t, exists := s[name]; exists {
				result = append(result, t)
			}
		}
		return result
	}

	// Sort alphabetically for deterministic output
	names := make([]string, 0, len(s))
	for name := range s {
		names = append(names, name)
	}
	sort.Strings(names)

	result := make([]Tool, 0, len(s))
	for _, name := range names {
		result = append(result, s[name])
	}
	return result
}

// ApplyToolOrder reorders an already-ordered tool slice so the tools named in
// toolOrder appear first in that order; tools not named keep their incoming
// relative order after them. A stable provider request shape lets providers
// reuse cached request prefixes. Mirrors ai-sdk's toolOrder option.
func ApplyToolOrder(tools []Tool, toolOrder []string) []Tool {
	if len(toolOrder) == 0 || len(tools) == 0 {
		return tools
	}
	byName := make(map[string]Tool, len(tools))
	for _, t := range tools {
		byName[t.Name] = t
	}
	result := make([]Tool, 0, len(tools))
	used := make(map[string]bool, len(toolOrder))
	for _, name := range toolOrder {
		if t, exists := byName[name]; exists && !used[name] {
			result = append(result, t)
			used[name] = true
		}
	}
	for _, t := range tools {
		if !used[t.Name] {
			result = append(result, t)
		}
	}
	return result
}
