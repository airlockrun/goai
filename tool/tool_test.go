package tool

import (
	"encoding/json"
	"reflect"
	"testing"
)

func names(tools []Tool) []string {
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Name
	}
	return out
}

func TestApplyToolOrder(t *testing.T) {
	in := []Tool{{Name: "alpha"}, {Name: "beta"}, {Name: "gamma"}}

	tests := []struct {
		name      string
		toolOrder []string
		want      []string
	}{
		{"empty order keeps input", nil, []string{"alpha", "beta", "gamma"}},
		{"full order", []string{"gamma", "alpha", "beta"}, []string{"gamma", "alpha", "beta"}},
		{"partial order, rest keep relative order", []string{"gamma"}, []string{"gamma", "alpha", "beta"}},
		{"unknown names ignored", []string{"missing", "beta"}, []string{"beta", "alpha", "gamma"}},
		{"duplicates ignored", []string{"beta", "beta"}, []string{"beta", "alpha", "gamma"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := names(ApplyToolOrder(in, tt.toolOrder))
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ApplyToolOrder(%v) = %v, want %v", tt.toolOrder, got, tt.want)
			}
		})
	}
}

func TestBuild_DefaultInputSchema(t *testing.T) {
	t.Run("no schema defaults to empty object", func(t *testing.T) {
		got := New("ping").Description("no input").Build()
		var s map[string]any
		if err := json.Unmarshal(got.InputSchema, &s); err != nil {
			t.Fatalf("default schema is not valid JSON: %v", err)
		}
		if s["type"] != "object" {
			t.Errorf("default schema type = %v, want object", s["type"])
		}
	})

	t.Run("explicit schema is preserved", func(t *testing.T) {
		explicit := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
		got := New("echo").Description("with input").Schema(explicit).Build()
		if string(got.InputSchema) != string(explicit) {
			t.Errorf("InputSchema = %s, want %s", got.InputSchema, explicit)
		}
	})
}
