package middleware

import "testing"

func TestGetPotentialStartIndex(t *testing.T) {
	cases := []struct {
		name     string
		text     string
		searched string
		want     int
	}{
		{"empty searched", "hello", "", -1},
		{"direct substring", "abc<think>def", "<think>", 3},
		{"partial suffix match", "hello <thi", "<think>", 6},
		{"no match", "hello world", "<think>", -1},
		{"full tag at end", "hello <think>", "<think>", 6},
		{"single char partial", "hello <", "<think>", 6},
		{"empty text", "", "<think>", -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := getPotentialStartIndex(tc.text, tc.searched); got != tc.want {
				t.Errorf("getPotentialStartIndex(%q, %q) = %d, want %d", tc.text, tc.searched, got, tc.want)
			}
		})
	}
}
