package anthropic

import (
	"regexp"
	"strings"
)

var oldClaude = regexp.MustCompile(`claude-(?:instant(?:-|$)|v?2(?:$|[-.:])|3(?:$|[-.]))`)

func modelSupport(id string) (limit int, known, rejectsSampling bool) {
	for _, name := range []string{"claude-opus-5", "claude-fable-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7"} {
		if strings.Contains(id, name) {
			return 128000, true, true
		}
	}
	for _, name := range []string{"claude-sonnet-4-6", "claude-opus-4-6"} {
		if strings.Contains(id, name) {
			return 128000, true, false
		}
	}
	for _, name := range []string{"claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5", "claude-sonnet-4"} {
		if strings.Contains(id, name) {
			return 64000, true, false
		}
	}
	if strings.Contains(id, "claude-opus-4") {
		return 32000, true, false
	}
	if strings.Contains(id, "claude-3-haiku") {
		return 4096, true, false
	}
	if oldClaude.MatchString(id) {
		return 4096, false, false
	}
	if strings.Contains(id, "claude-") {
		return 128000, false, true
	}
	return 4096, false, false
}
