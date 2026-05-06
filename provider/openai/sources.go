package openai

import (
	"crypto/rand"
	"encoding/hex"

	"github.com/airlockrun/goai/stream"
)

// annotationToSource maps an OpenAI Responses annotation to a
// stream.SourceEvent. Mirrors ai-sdk's url_citation / file_citation /
// container_file_citation / file_path handling in
// packages/openai/src/responses/openai-responses-language-model.ts.
//
// Returns ok=false for unknown annotation types so callers can skip them.
func annotationToSource(a responsesAnnotation) (stream.SourceEvent, bool) {
	switch a.Type {
	case "url_citation":
		return stream.SourceEvent{
			SourceType: stream.SourceTypeURL,
			ID:         newSourceID(),
			URL:        a.URL,
			Title:      a.Title,
		}, true
	case "file_citation":
		return stream.SourceEvent{
			SourceType: stream.SourceTypeDocument,
			ID:         newSourceID(),
			MediaType:  "text/plain",
			Title:      a.Filename,
			Filename:   a.Filename,
			ProviderMetadata: map[string]any{
				"openai": map[string]any{
					"type":   a.Type,
					"fileId": a.FileID,
					"index":  a.Index,
				},
			},
		}, true
	case "container_file_citation":
		return stream.SourceEvent{
			SourceType: stream.SourceTypeDocument,
			ID:         newSourceID(),
			MediaType:  "text/plain",
			Title:      a.Filename,
			Filename:   a.Filename,
			ProviderMetadata: map[string]any{
				"openai": map[string]any{
					"type":        a.Type,
					"fileId":      a.FileID,
					"containerId": a.ContainerID,
				},
			},
		}, true
	case "file_path":
		return stream.SourceEvent{
			SourceType: stream.SourceTypeDocument,
			ID:         newSourceID(),
			MediaType:  "application/octet-stream",
			Title:      a.FileID,
			Filename:   a.FileID,
			ProviderMetadata: map[string]any{
				"openai": map[string]any{
					"type":   a.Type,
					"fileId": a.FileID,
					"index":  a.Index,
				},
			},
		}, true
	}
	return stream.SourceEvent{}, false
}

// newSourceID returns an opaque random hex ID for a source. ai-sdk uses
// a configurable generator (defaults to ~16 hex chars); we match that
// length for parity. crypto/rand failure is unrecoverable here, so we
// panic — never expected outside of catastrophic system failure.
func newSourceID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		panic("goai/openai: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(buf[:])
}
