package stream

import (
	"encoding/json"
	"testing"
)

func TestWarningHelpers(t *testing.T) {
	tests := []struct {
		name string
		got  Warning
		want Warning
	}{
		{
			name: "unsupported with details",
			got:  UnsupportedWarning("frequencyPenalty", "not supported"),
			want: Warning{Type: WarningUnsupported, Feature: "frequencyPenalty", Details: "not supported"},
		},
		{
			name: "unsupported without details",
			got:  UnsupportedWarning("topK", ""),
			want: Warning{Type: WarningUnsupported, Feature: "topK"},
		},
		{
			name: "compatibility",
			got:  CompatibilityWarning("temperature", "clamped to 1.0"),
			want: Warning{Type: WarningCompatibility, Feature: "temperature", Details: "clamped to 1.0"},
		},
		{
			name: "other",
			got:  OtherWarning("something happened"),
			want: Warning{Type: WarningOther, Message: "something happened"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Errorf("got %+v, want %+v", tc.got, tc.want)
			}
		})
	}
}

func TestWarningJSONRoundTrip(t *testing.T) {
	tests := []struct {
		name    string
		warning Warning
		wantKey string
	}{
		{
			name:    "unsupported",
			warning: UnsupportedWarning("seed", "not supported"),
			wantKey: `"type":"unsupported"`,
		},
		{
			name:    "compatibility",
			warning: CompatibilityWarning("temperature", "clamped"),
			wantKey: `"type":"compatibility"`,
		},
		{
			name:    "other",
			warning: OtherWarning("info"),
			wantKey: `"type":"other"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.warning)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if !containsSubstring(string(b), tc.wantKey) {
				t.Errorf("marshal = %s, missing %s", b, tc.wantKey)
			}
			var round Warning
			if err := json.Unmarshal(b, &round); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if round != tc.warning {
				t.Errorf("round trip: got %+v, want %+v", round, tc.warning)
			}
		})
	}
}

func TestWarningOmitEmpty(t *testing.T) {
	w := Warning{Type: WarningUnsupported, Feature: "topK"}
	b, err := json.Marshal(w)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	if containsSubstring(s, `"message"`) {
		t.Errorf("empty message should be omitted: %s", s)
	}
	if containsSubstring(s, `"details"`) {
		t.Errorf("empty details should be omitted: %s", s)
	}
}

func TestStartEventWarningsOmitEmpty(t *testing.T) {
	b, err := json.Marshal(StartEvent{})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if containsSubstring(string(b), `"warnings"`) {
		t.Errorf("empty warnings should be omitted: %s", b)
	}

	withWarn := StartEvent{Warnings: []Warning{UnsupportedWarning("topK", "")}}
	b, err = json.Marshal(withWarn)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !containsSubstring(string(b), `"warnings"`) {
		t.Errorf("non-empty warnings should serialize: %s", b)
	}
}

func containsSubstring(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
