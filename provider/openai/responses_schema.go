package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"sort"
)

// Responses integer constraints must fit signed 64-bit bounds. Keep numbers as
// JSON numbers: float64 rounds the integer extrema outside their declared range.
// Constraint-only allOf branches inherit the sibling property's declared type.
func validateResponsesIntegerBounds(raw json.RawMessage) error {
	if len(raw) == 0 {
		return nil
	}
	var root map[string]any
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := d.Decode(&root); err != nil {
		return fmt.Errorf("parameters: invalid JSON schema: %w", err)
	}
	min, _ := new(big.Rat).SetString("-9223372036854775808")
	max, _ := new(big.Rat).SetString("9223372036854775807")
	var walk func(map[string]any, map[string]any, string) error
	walk = func(node, inherited map[string]any, path string) error {
		effective := make(map[string]any, len(inherited)+len(node))
		for k, v := range inherited {
			effective[k] = v
		}
		for k, v := range node {
			effective[k] = v
		}
		kind := effective["type"]
		integer := kind == "integer"
		if kinds, ok := kind.([]any); ok {
			for _, k := range kinds {
				integer = integer || k == "integer"
			}
		}
		if integer {
			for _, key := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} {
				if n, ok := node[key].(json.Number); ok {
					value, valid := new(big.Rat).SetString(string(n))
					if !valid || value.Cmp(min) < 0 || value.Cmp(max) > 0 {
						return fmt.Errorf("%s.%s: integer bound %s is outside signed 64-bit range [-9223372036854775808, 9223372036854775807]", path, key, n)
					}
				}
			}
		}
		for _, key := range []string{"properties", "patternProperties", "$defs", "definitions"} {
			children, _ := node[key].(map[string]any)
			base, _ := inherited[key].(map[string]any)
			keys := make([]string, 0, len(children))
			for name := range children {
				keys = append(keys, name)
			}
			sort.Strings(keys)
			for _, name := range keys {
				child, ok := children[name].(map[string]any)
				if !ok {
					continue
				}
				var parent map[string]any
				if key == "properties" || key == "patternProperties" {
					parent, _ = base[name].(map[string]any)
				}
				if err := walk(child, parent, path+"."+key+"."+name); err != nil {
					return err
				}
			}
		}
		for _, key := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
			children, _ := node[key].([]any)
			for i, value := range children {
				if child, ok := value.(map[string]any); ok {
					var parent map[string]any
					if key != "prefixItems" {
						parent = effective
					}
					if err := walk(child, parent, fmt.Sprintf("%s.%s[%d]", path, key, i)); err != nil {
						return err
					}
				}
			}
		}
		for _, key := range []string{"items", "additionalProperties", "not", "if", "then", "else", "contains"} {
			if child, ok := node[key].(map[string]any); ok {
				parent, _ := inherited[key].(map[string]any)
				if err := walk(child, parent, path+"."+key); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return walk(root, nil, "parameters")
}
