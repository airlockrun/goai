package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

func TestResponsesIntegerBounds(t *testing.T) {
	for _, tc := range []struct {
		name, schema, problem string
	}{
		{"int64", `{"type":"integer","minimum":-9223372036854775808,"maximum":9223372036854775807}`, ""},
		{"rounded maximum", `{"type":"integer","maximum":9223372036854776000}`, "parameters.maximum"},
		{"rounded minimum", `{"type":"integer","minimum":-9223372036854776000}`, "parameters.minimum"},
		{"float bounds", `{"type":"number","minimum":-1.7976931348623157e308,"maximum":1.7976931348623157e308}`, ""},
		{"fractional integer constraint", `{"type":"integer","minimum":0.5,"maximum":100.5}`, ""},
		{"allOf typed sibling", `{"type":"object","properties":{"count":{"type":"integer"}},"allOf":[{"properties":{"count":{"maximum":9223372036854776000}}}]}`, "parameters.allOf[0].properties.count.maximum"},
		{"nested definitions", `{"$defs":{"output":{"type":"object","properties":{"count":{"type":"integer"}},"allOf":[{"properties":{"count":{"minimum":-9223372036854776000}}}]}},"oneOf":[{"$ref":"#/$defs/output"}]}`, "parameters.$defs.output.allOf[0].properties.count.minimum"},
		{"nullable array item", `{"type":"array","items":{"type":["integer","null"],"exclusiveMaximum":9223372036854775808}}`, "parameters.items.exclusiveMaximum"},
		{"data annotation", `{"type":"object","default":{"maximum":9223372036854776000},"properties":{"maximum":{"type":"number","maximum":1e100}}}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateResponsesIntegerBounds(json.RawMessage(tc.schema))
			if tc.problem == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.problem) {
				t.Fatalf("error = %v, want path %s", err, tc.problem)
			}
		})
	}
}

func TestResponsesSchemaPreservesIntegerBounds(t *testing.T) {
	raw := json.RawMessage(`{"type":"object","$defs":{"output":{"type":"object","properties":{"count":{"type":"integer"}},"allOf":[{"properties":{"count":{"minimum":-9223372036854775808,"maximum":9223372036854775807}}}]}},"oneOf":[{"type":"object","properties":{"output":{"$ref":"#/$defs/output"}}}]}`)
	for _, strict := range []bool{false, true} {
		name := "non-strict"
		if strict {
			name = "strict"
		}
		t.Run(name, func(t *testing.T) {
			body, _, err := (&ResponsesModel{id: "test"}).buildRequest(&stream.CallOptions{Tools: []tool.Tool{{Name: "submit", InputSchema: raw}}, ProviderOptions: map[string]any{"strictJsonSchema": strict}})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(body, []byte(`"minimum":-9223372036854775808`)) || !bytes.Contains(body, []byte(`"maximum":9223372036854775807`)) {
				t.Fatalf("integer bounds changed: %s", body)
			}
		})
	}
}

func TestResponsesRejectsRoundedIntegerBounds(t *testing.T) {
	_, _, err := (&ResponsesModel{id: "test"}).buildRequest(&stream.CallOptions{Tools: []tool.Tool{{Name: "submit", InputSchema: json.RawMessage(`{"type":"integer","maximum":9223372036854776000}`)}}})
	if err == nil || !strings.Contains(err.Error(), `tool "submit"`) || !strings.Contains(err.Error(), "parameters.maximum") {
		t.Fatalf("error = %v, want tool name and bound path", err)
	}
}

func TestResponsesIntegerBoundsHTTPBoundary(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var request struct {
			Tools []struct {
				Parameters json.RawMessage `json:"parameters"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if len(request.Tools) != 1 || !bytes.Contains(request.Tools[0].Parameters, []byte(`"maximum":9223372036854775807`)) {
			t.Errorf("integer bound changed at HTTP boundary: %+v", request)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"test\",\"status\":\"completed\",\"output\":[]}}\n\n")
	}))
	defer server.Close()
	model := NewResponsesModel("test", ResponsesConfig{Provider: "openai.responses", URL: server.URL, HTTPClient: server.Client(), ConfigureRequest: func(*http.Request) error { return nil }})
	for _, tc := range []struct {
		maximum string
		valid   bool
	}{{"9223372036854775807", true}, {"9223372036854776000", false}} {
		before := requests
		events, err := model.Stream(t.Context(), &stream.CallOptions{Tools: []tool.Tool{{Name: "submit", InputSchema: json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer","maximum":` + tc.maximum + `}}}`)}}, ProviderOptions: map[string]any{"strictJsonSchema": true}})
		if err != nil {
			t.Fatal(err)
		}
		failed := false
		for event := range events {
			if event.Type == stream.EventError {
				failed = true
			}
		}
		if failed == tc.valid || (requests != before) != tc.valid {
			t.Fatalf("maximum=%s failed=%v requests=%d", tc.maximum, failed, requests-before)
		}
	}
}
