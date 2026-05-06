package xai

import (
	"crypto/rand"
	"encoding/hex"

	"github.com/airlockrun/goai/stream"
)

// annotationToSource maps an xAI url_citation annotation to a
// stream.SourceEvent. Mirrors ai-sdk's
// packages/xai/src/responses/xai-responses-language-model.ts. Returns
// ok=false for non-url_citation types so callers can skip them.
func annotationToSource(a responsesAnnotation) (stream.SourceEvent, bool) {
	if a.Type != "url_citation" || a.URL == "" {
		return stream.SourceEvent{}, false
	}
	title := a.Title
	if title == "" {
		title = a.URL
	}
	return stream.SourceEvent{
		SourceType: stream.SourceTypeURL,
		ID:         newSourceID(),
		URL:        a.URL,
		Title:      title,
	}, true
}

func newSourceID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		panic("goai/xai: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(buf[:])
}
