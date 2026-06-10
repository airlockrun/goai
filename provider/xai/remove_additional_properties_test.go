package xai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestSanitizeToolSchema(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "strips top-level additionalProperties false",
			in:   `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}`,
			want: `{"type":"object","properties":{"a":{"type":"string"}}}`,
		},
		{
			name: "strips nested additionalProperties false",
			in:   `{"type":"object","properties":{"address":{"type":"object","properties":{"city":{"type":"string"}},"additionalProperties":false}},"additionalProperties":false}`,
			want: `{"type":"object","properties":{"address":{"type":"object","properties":{"city":{"type":"string"}}}}}`,
		},
		{
			name: "keeps additionalProperties schema object",
			in:   `{"type":"object","additionalProperties":{"type":"string"}}`,
			want: `{"type":"object","additionalProperties":{"type":"string"}}`,
		},
		{
			name: "keeps additionalProperties true",
			in:   `{"type":"object","additionalProperties":true}`,
			want: `{"type":"object","additionalProperties":true}`,
		},
		{
			name: "does not strip a property literally named properties",
			in:   `{"type":"object","properties":{"properties":{"type":"object","properties":{"city":{"type":"string"}},"additionalProperties":false}},"additionalProperties":false}`,
			want: `{"type":"object","properties":{"properties":{"type":"object","properties":{"city":{"type":"string"}}}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeToolSchema(json.RawMessage(tt.in))
			var gotMap, wantMap any
			if err := json.Unmarshal(got, &gotMap); err != nil {
				t.Fatalf("unmarshal got: %v", err)
			}
			if err := json.Unmarshal([]byte(tt.want), &wantMap); err != nil {
				t.Fatalf("unmarshal want: %v", err)
			}
			if !reflect.DeepEqual(gotMap, wantMap) {
				t.Errorf("sanitizeToolSchema:\n got  %s\n want %s", got, tt.want)
			}
		})
	}

	t.Run("empty schema passes through", func(t *testing.T) {
		if got := sanitizeToolSchema(nil); got != nil {
			t.Errorf("nil schema: got %q, want nil", got)
		}
	})
}
