package tool

import (
	"context"
	"encoding/json"
	"testing"
)

type calcIn struct {
	Expression string `json:"expression"`
}
type calcOut struct {
	Result int `json:"result"`
}

func TestTyped_SchemasAndExecute(t *testing.T) {
	tl := Typed[calcIn, calcOut]("calc").
		Description("evaluate").
		Execute(func(_ context.Context, in calcIn) (calcOut, error) {
			return calcOut{Result: len(in.Expression)}, nil
		}).
		Build()

	if tl.Name != "calc" || tl.Description != "evaluate" {
		t.Fatalf("name/desc = %q/%q", tl.Name, tl.Description)
	}
	if typeOf(t, tl.InputSchema) != "object" {
		t.Errorf("input schema not object: %s", tl.InputSchema)
	}
	if len(tl.OutputSchema) == 0 {
		t.Fatal("output schema missing")
	}
	if typeOf(t, tl.OutputSchema) != "object" {
		t.Errorf("output schema not object: %s", tl.OutputSchema)
	}

	res, err := tl.Execute(context.Background(), json.RawMessage(`{"expression":"abc"}`), CallOptions{})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if res.Output != `{"result":3}` {
		t.Errorf("output = %q, want {\"result\":3}", res.Output)
	}
}

// A no-argument tool's In reflects to a non-object schema; Build must coerce the
// advertised input schema to an object so strict validators accept it.
func TestTyped_NoArgInputIsObject(t *testing.T) {
	tl := Typed[any, string]("ping").
		Description("ping").
		Execute(func(_ context.Context, _ any) (string, error) { return "pong", nil }).
		Build()

	if typeOf(t, tl.InputSchema) != "object" {
		t.Errorf("no-arg input schema must be object, got: %s", tl.InputSchema)
	}
	res, err := tl.Execute(context.Background(), nil, CallOptions{})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if res.Output != `"pong"` {
		t.Errorf("output = %q, want \"pong\"", res.Output)
	}
}

func typeOf(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("schema not an object: %s (%v)", raw, err)
	}
	s, _ := m["type"].(string)
	return s
}
