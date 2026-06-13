package message

import (
	"encoding/json"
	"errors"
	"fmt"
)

// FileData is the tagged-union payload of a FilePart: the concrete variant
// says how the file bytes are carried. JSON encodes the variant as the nested
// "type" field. Mirrors ai-sdk's file part data union
// (references/ai-sdk/packages/provider-utils/src/types/content-part.ts).
type FileData interface {
	fileDataType() string
}

// FileDataBytes carries inline base64-encoded bytes (type:"data").
type FileDataBytes struct {
	Data string `json:"data"` // base64-encoded
}

func (FileDataBytes) fileDataType() string { return "data" }

// FileDataURL points to a remote file the provider fetches (type:"url").
type FileDataURL struct {
	URL string `json:"url"`
}

func (FileDataURL) fileDataType() string { return "url" }

// FileDataReference is a provider-side handle to a previously uploaded file
// (type:"reference"), e.g. an OpenAI/Anthropic Files API id.
type FileDataReference struct {
	Reference map[string]any `json:"reference"`
}

func (FileDataReference) fileDataType() string { return "reference" }

// FileDataText carries the file content as text (type:"text").
type FileDataText struct {
	Text string `json:"text"`
}

func (FileDataText) fileDataType() string { return "text" }

// marshalFileData serializes a FileData with its "type" discriminant injected
// as the leading field.
func marshalFileData(d FileData) (json.RawMessage, error) {
	if d == nil {
		return nil, errors.New("file part: nil data")
	}
	inner, err := json.Marshal(d)
	if err != nil {
		return nil, err
	}
	return injectType(d.fileDataType(), inner), nil
}

// unmarshalFileData decodes a FileData keyed by its "type" field.
func unmarshalFileData(raw json.RawMessage) (FileData, error) {
	var peek struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &peek); err != nil {
		return nil, fmt.Errorf("invalid file data: %w", err)
	}
	switch peek.Type {
	case "data":
		var d FileDataBytes
		if err := json.Unmarshal(raw, &d); err != nil {
			return nil, err
		}
		return d, nil
	case "url":
		var d FileDataURL
		if err := json.Unmarshal(raw, &d); err != nil {
			return nil, err
		}
		return d, nil
	case "reference":
		var d FileDataReference
		if err := json.Unmarshal(raw, &d); err != nil {
			return nil, err
		}
		return d, nil
	case "text":
		var d FileDataText
		if err := json.Unmarshal(raw, &d); err != nil {
			return nil, err
		}
		return d, nil
	default:
		return nil, fmt.Errorf("unknown file data type: %q", peek.Type)
	}
}
