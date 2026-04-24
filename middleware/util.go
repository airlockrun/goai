package middleware

import "strings"

// getPotentialStartIndex returns the earliest index in `text` where a prefix
// (up to and including all) of `searchedText` begins, or -1 if no such index
// exists. Used by extractReasoning to detect partially-streamed tags — the
// tag may arrive split across multiple stream deltas, so we need to spot a
// potential match even when only its leading characters are present.
//
// Mirrors ai-sdk's getPotentialStartIndex
// (packages/ai/src/util/get-potential-start-index.ts), except the sentinel
// is -1 instead of null to match Go's strings.Index convention.
func getPotentialStartIndex(text, searchedText string) int {
	if searchedText == "" {
		return -1
	}
	if idx := strings.Index(text, searchedText); idx >= 0 {
		return idx
	}
	// Largest suffix of text that matches a prefix of searchedText.
	for i := 0; i < len(text); i++ {
		if strings.HasPrefix(searchedText, text[i:]) {
			return i
		}
	}
	return -1
}
