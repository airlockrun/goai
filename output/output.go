// Package output provides output strategies for structured text generation.
// This mirrors the ai-sdk output module (text, object, array, choice, json).
//
// All strategies implement stream.Output so they can be assigned to
// stream.Input.Output.
package output

import (
	"encoding/json"
	"fmt"

	"github.com/airlockrun/goai/schema"
	"github.com/airlockrun/goai/stream"
)

// Text generates plain text.
// This is the default output mode.
func Text() stream.Output {
	return &textOutput{}
}

type textOutput struct{}

func (t *textOutput) Name() string { return "text" }

func (t *textOutput) ResponseFormat() *stream.ResponseFormat {
	return &stream.ResponseFormat{Type: "text"}
}

func (t *textOutput) ParseComplete(text string, ctx stream.OutputParseContext) (any, error) {
	return text, nil
}

func (t *textOutput) ParsePartial(text string) any {
	return text
}

// ObjectOptions configures object output generation.
type ObjectOptions struct {
	Schema      *schema.Schema
	Name        string
	Description string
}

// Object generates a typed object matching the schema.
func Object(opts ObjectOptions) stream.Output {
	return &objectOutput{opts: opts}
}

type objectOutput struct {
	opts ObjectOptions
}

func (o *objectOutput) Name() string { return "object" }

func (o *objectOutput) ResponseFormat() *stream.ResponseFormat {
	schemaJSON, _ := o.opts.Schema.JSON()
	return &stream.ResponseFormat{
		Type:        "json",
		Schema:      schemaJSON,
		Name:        o.opts.Name,
		Description: o.opts.Description,
	}
}

func (o *objectOutput) ParseComplete(text string, ctx stream.OutputParseContext) (any, error) {
	var result any
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, &NoObjectGeneratedError{
			Message:      "No object generated: could not parse the response.",
			Cause:        err,
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}

	// TODO: Add schema validation
	return result, nil
}

func (o *objectOutput) ParsePartial(text string) any {
	var result any
	// Try to parse partial JSON (may fail)
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil
	}
	return result
}

// ArrayOptions configures array output generation.
type ArrayOptions struct {
	// Element is the schema for array elements.
	Element     *schema.Schema
	Name        string
	Description string
}

// Array generates an array of elements matching the element schema.
// The model responds with {"elements": [...]} wrapper.
func Array(opts ArrayOptions) stream.Output {
	return &arrayOutput{opts: opts}
}

type arrayOutput struct {
	opts ArrayOptions
}

func (a *arrayOutput) Name() string { return "array" }

func (a *arrayOutput) ResponseFormat() *stream.ResponseFormat {
	elementJSON, _ := a.opts.Element.JSON()

	// Wrap in elements array schema
	wrapperSchema := map[string]any{
		"$schema": "http://json-schema.org/draft-07/schema#",
		"type":    "object",
		"properties": map[string]any{
			"elements": map[string]any{
				"type":  "array",
				"items": json.RawMessage(elementJSON),
			},
		},
		"required":             []string{"elements"},
		"additionalProperties": false,
	}
	schemaJSON, _ := json.Marshal(wrapperSchema)

	return &stream.ResponseFormat{
		Type:        "json",
		Schema:      schemaJSON,
		Name:        a.opts.Name,
		Description: a.opts.Description,
	}
}

func (a *arrayOutput) ParseComplete(text string, ctx stream.OutputParseContext) (any, error) {
	var wrapper struct {
		Elements []any `json:"elements"`
	}
	if err := json.Unmarshal([]byte(text), &wrapper); err != nil {
		return nil, &NoObjectGeneratedError{
			Message:      "No object generated: could not parse the response.",
			Cause:        err,
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}

	if wrapper.Elements == nil {
		return nil, &NoObjectGeneratedError{
			Message:      "No object generated: response must contain elements array.",
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}

	// TODO: Validate each element against schema
	return wrapper.Elements, nil
}

func (a *arrayOutput) ParsePartial(text string) any {
	var wrapper struct {
		Elements []any `json:"elements"`
	}
	if err := json.Unmarshal([]byte(text), &wrapper); err != nil {
		return nil
	}
	return wrapper.Elements
}

// ChoiceOptions configures choice output generation.
type ChoiceOptions struct {
	Options     []string
	Name        string
	Description string
}

// Choice generates one of the predefined choice options.
// The model responds with {"result": "chosen_option"} wrapper.
func Choice(opts ChoiceOptions) stream.Output {
	return &choiceOutput{opts: opts}
}

type choiceOutput struct {
	opts ChoiceOptions
}

func (c *choiceOutput) Name() string { return "choice" }

func (c *choiceOutput) ResponseFormat() *stream.ResponseFormat {
	choiceSchema := map[string]any{
		"$schema": "http://json-schema.org/draft-07/schema#",
		"type":    "object",
		"properties": map[string]any{
			"result": map[string]any{
				"type": "string",
				"enum": c.opts.Options,
			},
		},
		"required":             []string{"result"},
		"additionalProperties": false,
	}
	schemaJSON, _ := json.Marshal(choiceSchema)

	return &stream.ResponseFormat{
		Type:        "json",
		Schema:      schemaJSON,
		Name:        c.opts.Name,
		Description: c.opts.Description,
	}
}

func (c *choiceOutput) ParseComplete(text string, ctx stream.OutputParseContext) (any, error) {
	var wrapper struct {
		Result string `json:"result"`
	}
	if err := json.Unmarshal([]byte(text), &wrapper); err != nil {
		return nil, &NoObjectGeneratedError{
			Message:      "No object generated: could not parse the response.",
			Cause:        err,
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}

	// Validate choice is in options
	valid := false
	for _, opt := range c.opts.Options {
		if opt == wrapper.Result {
			valid = true
			break
		}
	}

	if !valid {
		return nil, &NoObjectGeneratedError{
			Message:      fmt.Sprintf("No object generated: '%s' is not a valid choice.", wrapper.Result),
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}

	return wrapper.Result, nil
}

func (c *choiceOutput) ParsePartial(text string) any {
	var wrapper struct {
		Result string `json:"result"`
	}
	if err := json.Unmarshal([]byte(text), &wrapper); err != nil {
		return nil
	}
	return wrapper.Result
}

// JSONOptions configures unstructured JSON output generation.
type JSONOptions struct {
	Name        string
	Description string
}

// JSON generates unstructured JSON (any valid JSON value).
func JSON(opts JSONOptions) stream.Output {
	return &jsonOutput{opts: opts}
}

type jsonOutput struct {
	opts JSONOptions
}

func (j *jsonOutput) Name() string { return "json" }

func (j *jsonOutput) ResponseFormat() *stream.ResponseFormat {
	return &stream.ResponseFormat{
		Type:        "json",
		Name:        j.opts.Name,
		Description: j.opts.Description,
	}
}

func (j *jsonOutput) ParseComplete(text string, ctx stream.OutputParseContext) (any, error) {
	var result any
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, &NoObjectGeneratedError{
			Message:      "No object generated: could not parse the response.",
			Cause:        err,
			Text:         text,
			FinishReason: ctx.FinishReason,
			Usage:        ctx.Usage,
		}
	}
	return result, nil
}

func (j *jsonOutput) ParsePartial(text string) any {
	var result any
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil
	}
	return result
}

// NoObjectGeneratedError indicates the model failed to generate a valid object.
type NoObjectGeneratedError struct {
	Message      string
	Cause        error
	Text         string
	FinishReason stream.FinishReason
	Usage        stream.Usage
}

func (e *NoObjectGeneratedError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

func (e *NoObjectGeneratedError) Unwrap() error {
	return e.Cause
}
