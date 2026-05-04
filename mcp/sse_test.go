package mcp

import (
	"strings"
	"testing"
)

func TestScanSSE(t *testing.T) {
	type event struct{ typ, data, id string }

	tests := []struct {
		name string
		in   string
		want []event
	}{
		{
			name: "single message default type",
			in:   "data: hello\n\n",
			want: []event{{"message", "hello", ""}},
		},
		{
			name: "explicit event type",
			in:   "event: endpoint\ndata: /messages/abc\n\n",
			want: []event{{"endpoint", "/messages/abc", ""}},
		},
		{
			name: "multi-line data joined with newline",
			in:   "data: line1\ndata: line2\n\n",
			want: []event{{"message", "line1\nline2", ""}},
		},
		{
			name: "id field captured",
			in:   "id: 42\nevent: message\ndata: payload\n\n",
			want: []event{{"message", "payload", "42"}},
		},
		{
			name: "comment lines skipped",
			in:   ": ping\ndata: hi\n\n",
			want: []event{{"message", "hi", ""}},
		},
		{
			name: "two events back to back",
			in:   "event: endpoint\ndata: /m\n\nevent: message\ndata: {\"x\":1}\n\n",
			want: []event{
				{"endpoint", "/m", ""},
				{"message", `{"x":1}`, ""},
			},
		},
		{
			name: "trailing event without blank line is dropped",
			in:   "data: orphan\n",
			want: nil,
		},
		{
			name: "value with no leading space preserved",
			in:   "data:value\n\n",
			want: []event{{"message", "value", ""}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []event
			err := scanSSE(strings.NewReader(tt.in), func(typ, data, id string) {
				got = append(got, event{typ, data, id})
			})
			if err != nil {
				t.Fatalf("scanSSE error: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %d events, want %d: %#v", len(got), len(tt.want), got)
			}
			for i, e := range got {
				if e != tt.want[i] {
					t.Errorf("event %d: got %#v, want %#v", i, e, tt.want[i])
				}
			}
		})
	}
}
