package message

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestToolContentParts(t *testing.T) {
	options := map[string]any{"provider": map[string]any{"cache": true}}
	output := ContentOutput{Value: []ToolContentItem{
		{Type: "text", Text: "before", ProviderOptions: options},
		{Type: "file-data", Data: "cGRm", MediaType: "application/pdf", Filename: "report.pdf", ProviderOptions: options},
		{Type: "text", Text: "after"},
		{Type: "image-url", URL: "https://example.invalid/image"},
		{Type: "image-url", URL: "data:image/png;base64,aW1n"},
	}}
	before, _ := json.Marshal(output)
	parts, err := ToolContentParts(output)
	if err != nil {
		t.Fatal(err)
	}
	want := []Part{
		TextPart{Text: "before", ProviderOptions: options},
		FilePart{Data: FileDataBytes{Data: "cGRm"}, MimeType: "application/pdf", Filename: "report.pdf", ProviderOptions: options},
		TextPart{Text: "after"},
		FilePart{Data: FileDataURL{URL: "https://example.invalid/image"}, MimeType: "image/*"},
		FilePart{Data: FileDataBytes{Data: "aW1n"}, MimeType: "image/png"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("normalization mismatch: %#v", parts)
	}
	after, _ := json.Marshal(output)
	if string(before) != string(after) {
		t.Fatal("normalization mutated history")
	}
	if parts, err := ToolContentParts(JSONOutput{Value: map[string]any{"answer": 42}}); err != nil || parts != nil {
		t.Fatal("scalar outputs must not normalize to multipart")
	}
}

func TestToolContentPartsRejectsUnknown(t *testing.T) {
	for _, item := range []ToolContentItem{
		{Type: "private-kind", Data: "PRIVATE"},
		{Type: "file-reference", ProviderReference: map[string]string{"provider": "PRIVATE"}},
		{Type: "image-data", MediaType: "application/PRIVATE", Data: "PRIVATE"},
		{Type: "image-url", MediaType: "image/png", URL: "data:application/PRIVATE;base64,PRIVATE"},
	} {
		t.Run(item.Type, func(t *testing.T) {
			parts, err := ToolContentParts(ContentOutput{Value: []ToolContentItem{{Type: "text", Text: "before"}, item}})
			if err == nil || parts != nil || strings.Contains(err.Error(), "PRIVATE") || strings.Contains(err.Error(), item.Type) {
				t.Fatalf("expected static fail-closed error, got %v", err)
			}
		})
	}
}

func TestToolMessageResultsFinalCorrelationAndImmutability(t *testing.T) {
	output := ContentOutput{Value: []ToolContentItem{{Type: "text", Text: "original"}}}
	msg := NewToolMessage("first", "first-tool", TextOutput{Value: "one"})
	msg.Content.Parts = append(msg.Content.Parts, ToolResultPart{ToolCallID: "last", ToolName: "last-tool", Output: output}, TextPart{Text: "left"}, FilePart{MimeType: "image/png", Data: FileDataBytes{Data: "c3BlY2ltZW4="}, Filename: "image.png"}, TextPart{Text: "right"})
	before, _ := json.Marshal(msg)
	results, err := ToolMessageResults(msg)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].ToolCallID != "first" || results[0].Output.(TextOutput).Value != "one" || results[1].ToolCallID != "last" || results[1].ToolName != "last-tool" {
		t.Fatal("real results changed")
	}
	items := results[1].Output.(ContentOutput).Value
	if len(items) != 4 || items[0].Text != "original" || items[1].Text != "left" || items[2].Data != "c3BlY2ltZW4=" || items[2].Filename != "image.png" || items[3].Text != "right" {
		t.Fatal("attachment representation/order changed")
	}
	items[0].Text = "changed-result-copy"
	after, _ := json.Marshal(msg)
	if string(before) != string(after) {
		t.Fatal("normalization aliases persisted output slice")
	}
	plain := NewToolMessage("only", "tool", output)
	results, err = ToolMessageResults(plain)
	if err != nil {
		t.Fatal(err)
	}
	results[0].Output.(ContentOutput).Value[0].Text = "changed-without-attachments"
	if plain.Content.Parts[0].(ToolResultPart).Output.(ContentOutput).Value[0].Text != "original" {
		t.Fatal("result without attachments aliases persisted output slice")
	}
}
