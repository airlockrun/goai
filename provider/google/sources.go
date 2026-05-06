package google

import (
	"crypto/rand"
	"encoding/hex"
	"strings"

	"github.com/airlockrun/goai/stream"
)

// extractSources walks groundingMetadata.groundingChunks and returns a
// SourceEvent per chunk. Mirrors ai-sdk's extractSources at
// packages/google/src/google-language-model.ts. Web chunks become URL
// sources; retrievedContext chunks are URL sources when the URI is
// http(s) and document sources otherwise; maps chunks become URL
// sources keyed on the chunk URI.
func extractSources(g *geminiGroundingMetadata) []stream.SourceEvent {
	if g == nil || len(g.GroundingChunks) == 0 {
		return nil
	}
	out := make([]stream.SourceEvent, 0, len(g.GroundingChunks))
	for _, c := range g.GroundingChunks {
		switch {
		case c.Web != nil && c.Web.URI != "":
			out = append(out, stream.SourceEvent{
				SourceType: stream.SourceTypeURL,
				ID:         newSourceID(),
				URL:        c.Web.URI,
				Title:      c.Web.Title,
			})
		case c.RetrievedContext != nil:
			out = append(out, retrievedContextSource(c.RetrievedContext))
		case c.Maps != nil && c.Maps.URI != "":
			out = append(out, stream.SourceEvent{
				SourceType: stream.SourceTypeURL,
				ID:         newSourceID(),
				URL:        c.Maps.URI,
				Title:      c.Maps.Title,
			})
		}
	}
	return out
}

func retrievedContextSource(rc *geminiRetrievedContext) stream.SourceEvent {
	uri := rc.URI
	if uri != "" && (strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://")) {
		return stream.SourceEvent{
			SourceType: stream.SourceTypeURL,
			ID:         newSourceID(),
			URL:        uri,
			Title:      rc.Title,
		}
	}

	title := rc.Title
	if title == "" {
		title = "Unknown Document"
	}
	mediaType := "application/octet-stream"
	var filename string
	if uri != "" {
		filename = lastPathSegment(uri)
		switch {
		case strings.HasSuffix(uri, ".pdf"):
			mediaType = "application/pdf"
		case strings.HasSuffix(uri, ".txt"):
			mediaType = "text/plain"
		case strings.HasSuffix(uri, ".docx"):
			mediaType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
		case strings.HasSuffix(uri, ".doc"):
			mediaType = "application/msword"
		case strings.HasSuffix(uri, ".md"), strings.HasSuffix(uri, ".markdown"):
			mediaType = "text/markdown"
		}
	} else if rc.FileSearchStore != "" {
		filename = lastPathSegment(rc.FileSearchStore)
	}
	return stream.SourceEvent{
		SourceType: stream.SourceTypeDocument,
		ID:         newSourceID(),
		MediaType:  mediaType,
		Title:      title,
		Filename:   filename,
	}
}

func lastPathSegment(s string) string {
	if idx := strings.LastIndex(s, "/"); idx >= 0 {
		return s[idx+1:]
	}
	return s
}

func newSourceID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		panic("goai/google: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(buf[:])
}
