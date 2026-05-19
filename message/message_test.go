package message

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestContentJSON_TextOnly(t *testing.T) {
	msg := Message{
		Role:    RoleUser,
		Content: Content{Text: "hello"},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	// Text-only should serialize as {"role":"user","content":"hello"}
	expected := `{"role":"user","content":"hello"}`
	if string(data) != expected {
		t.Errorf("got %s, want %s", data, expected)
	}

	// Round-trip
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Role != msg.Role || got.Content.Text != msg.Content.Text {
		t.Errorf("round-trip mismatch: got %+v", got)
	}
	if got.Content.IsMultiPart() {
		t.Error("text-only content should not be multi-part after round-trip")
	}
}

func TestContentJSON_EmptyText(t *testing.T) {
	msg := Message{
		Role:    RoleSystem,
		Content: Content{Text: ""},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	expected := `{"role":"system","content":""}`
	if string(data) != expected {
		t.Errorf("got %s, want %s", data, expected)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Content.Text != "" {
		t.Errorf("expected empty text, got %q", got.Content.Text)
	}
}

func TestContentJSON_TextPart(t *testing.T) {
	msg := NewAssistantMessageWithParts(TextPart{Text: "hello world"})

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if !got.Content.IsMultiPart() {
		t.Fatal("expected multi-part")
	}
	if len(got.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(got.Content.Parts))
	}
	tp, ok := got.Content.Parts[0].(TextPart)
	if !ok {
		t.Fatalf("expected TextPart, got %T", got.Content.Parts[0])
	}
	if tp.Text != "hello world" {
		t.Errorf("got %q, want %q", tp.Text, "hello world")
	}
}

func TestContentJSON_ToolCallPart(t *testing.T) {
	msg := NewAssistantMessageWithParts(
		TextPart{Text: "I'll read the file"},
		ToolCallPart{
			ID:    "call_123",
			Name:  "read",
			Input: json.RawMessage(`{"path":"/tmp/test.txt"}`),
		},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}

	tp := got.Content.Parts[0].(TextPart)
	if tp.Text != "I'll read the file" {
		t.Errorf("text = %q", tp.Text)
	}

	tc := got.Content.Parts[1].(ToolCallPart)
	if tc.ID != "call_123" || tc.Name != "read" {
		t.Errorf("tool call = %+v", tc)
	}
	if string(tc.Input) != `{"path":"/tmp/test.txt"}` {
		t.Errorf("input = %s", tc.Input)
	}
}

func TestContentJSON_ToolResultPart(t *testing.T) {
	msg := Message{
		Role: RoleTool,
		Content: Content{Parts: []Part{
			ToolResultPart{
				ToolCallID: "call_123",
				ToolName:   "read",
				Output:     TextOutput{Value: "file contents here"},
			},
		}},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	tr := got.Content.Parts[0].(ToolResultPart)
	if tr.ToolCallID != "call_123" || tr.ToolName != "read" {
		t.Errorf("tool result = %+v", tr)
	}
	if ToolOutputText(tr.Output) != "file contents here" {
		t.Errorf("result = %v", tr.Output)
	}
}

func TestContentJSON_ReasoningPart(t *testing.T) {
	msg := NewAssistantMessageWithParts(
		ReasoningPart{
			Text:            "let me think...",
			ProviderOptions: map[string]any{"itemId": "item_1"},
		},
		TextPart{Text: "here's my answer"},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}

	rp := got.Content.Parts[0].(ReasoningPart)
	if rp.Text != "let me think..." {
		t.Errorf("reasoning text = %q", rp.Text)
	}
	if rp.ProviderOptions["itemId"] != "item_1" {
		t.Errorf("provider options = %v", rp.ProviderOptions)
	}
}

func TestContentJSON_ImagePart(t *testing.T) {
	msg := NewUserMessageWithParts(
		ImagePart{Image: "base64data", MimeType: "image/png"},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	ip := got.Content.Parts[0].(ImagePart)
	if ip.Image != "base64data" || ip.MimeType != "image/png" {
		t.Errorf("image part = %+v", ip)
	}
}

func TestContentJSON_FilePart(t *testing.T) {
	msg := NewUserMessageWithParts(
		FilePart{Data: "base64pdf", MimeType: "application/pdf", Filename: "doc.pdf"},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	fp := got.Content.Parts[0].(FilePart)
	if fp.Data != "base64pdf" || fp.MimeType != "application/pdf" || fp.Filename != "doc.pdf" {
		t.Errorf("file part = %+v", fp)
	}
}

func TestContentJSON_ToolApprovalRequestPart(t *testing.T) {
	msg := NewAssistantMessageWithParts(
		ToolApprovalRequestPart{
			ApprovalID: "apr_1",
			ToolCallID: "call_1",
			ToolName:   "bash",
			Input:      map[string]string{"command": "rm -rf /"},
		},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	ap := got.Content.Parts[0].(ToolApprovalRequestPart)
	if ap.ApprovalID != "apr_1" || ap.ToolCallID != "call_1" || ap.ToolName != "bash" {
		t.Errorf("approval request = %+v", ap)
	}
}

func TestContentJSON_ToolApprovalResponsePart(t *testing.T) {
	msg := NewUserMessageWithParts(
		ToolApprovalResponsePart{
			ApprovalID: "apr_1",
			Approved:   true,
			Reason:     "user approved",
		},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	ar := got.Content.Parts[0].(ToolApprovalResponsePart)
	if ar.ApprovalID != "apr_1" || !ar.Approved || ar.Reason != "user approved" {
		t.Errorf("approval response = %+v", ar)
	}
}

func TestContentJSON_MessageSliceRoundTrip(t *testing.T) {
	messages := []Message{
		NewSystemMessage("you are helpful"),
		NewUserMessage("hello"),
		NewAssistantMessageWithParts(
			ReasoningPart{Text: "thinking..."},
			TextPart{Text: "I'll help"},
			ToolCallPart{ID: "c1", Name: "read", Input: json.RawMessage(`{"path":"f.txt"}`)},
		),
		NewToolResultText("c1", "read", "file content"),
		NewAssistantMessage("done"),
	}

	data, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}

	var got []Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if len(got) != len(messages) {
		t.Fatalf("expected %d messages, got %d", len(messages), len(got))
	}

	// Verify each message
	if got[0].Role != RoleSystem || got[0].Content.Text != "you are helpful" {
		t.Errorf("msg[0] = %+v", got[0])
	}
	if got[1].Role != RoleUser || got[1].Content.Text != "hello" {
		t.Errorf("msg[1] = %+v", got[1])
	}
	if got[2].Role != RoleAssistant || len(got[2].Content.Parts) != 3 {
		t.Errorf("msg[2] parts = %d", len(got[2].Content.Parts))
	}
	if got[3].Role != RoleTool || len(got[3].Content.Parts) != 1 {
		t.Errorf("msg[3] = %+v", got[3])
	}
	if got[4].Role != RoleAssistant || got[4].Content.Text != "done" {
		t.Errorf("msg[4] = %+v", got[4])
	}
}

func TestContentJSON_UnknownType(t *testing.T) {
	data := `[{"type":"unknown","data":"something"}]`
	var c Content
	err := c.UnmarshalJSON([]byte(data))
	if err == nil {
		t.Error("expected error for unknown type")
	}
}

func TestContentJSON_InvalidJSON(t *testing.T) {
	var c Content
	err := c.UnmarshalJSON([]byte(`{invalid`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// TestToolResultPart_OutputRoundTrip verifies every ToolResultOutput variant
// survives a ToolResultPart JSON marshal→unmarshal: the concrete Go type and
// its value are preserved, and the serialized JSON carries the discriminated
// "output":{"type":"<variant>"} envelope (ADR §8).
func TestToolResultPart_OutputRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		output   ToolResultOutput
		wantType string
	}{
		{"text", TextOutput{Value: "hello"}, "text"},
		{"json", JSONOutput{Value: map[string]any{"k": "v"}}, "json"},
		{"error-text", ErrorTextOutput{Value: "boom"}, "error-text"},
		{"error-json", ErrorJSONOutput{Value: map[string]any{"code": float64(42)}}, "error-json"},
		{"execution-denied", ExecutionDeniedOutput{Reason: "nope"}, "execution-denied"},
		{"content", ContentOutput{Value: []ToolContentItem{
			{Type: "text", Text: "see image"},
			{Type: "image-data", Data: "Zm9v", MediaType: "image/png"},
		}}, "content"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			part := ToolResultPart{ToolCallID: "c1", ToolName: "tool", Output: tt.output}
			data, err := json.Marshal(part)
			if err != nil {
				t.Fatal(err)
			}
			wantEnvelope := `"output":{"type":"` + tt.wantType + `"`
			if !strings.Contains(string(data), wantEnvelope) {
				t.Errorf("marshaled JSON %s missing %s", data, wantEnvelope)
			}
			var got ToolResultPart
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if got.ToolCallID != "c1" || got.ToolName != "tool" {
				t.Errorf("identity lost: %+v", got)
			}
			if reflect.TypeOf(got.Output) != reflect.TypeOf(tt.output) {
				t.Fatalf("concrete type changed: got %T, want %T", got.Output, tt.output)
			}
			if got.Output.outputType() != tt.wantType {
				t.Errorf("outputType = %q, want %q", got.Output.outputType(), tt.wantType)
			}
			if !reflect.DeepEqual(got.Output, tt.output) {
				t.Errorf("value not preserved: got %#v, want %#v", got.Output, tt.output)
			}
		})
	}
}

// TestContentJSON_ToolApprovalParts_RoundTrip verifies tool-approval-request
// and tool-approval-response parts round-trip through Content (un)marshal
// (ADR §8).
func TestContentJSON_ToolApprovalParts_RoundTrip(t *testing.T) {
	msg := NewAssistantMessageWithParts(
		ToolApprovalRequestPart{
			ApprovalID: "apr_1",
			ToolCallID: "call_1",
			ToolName:   "bash",
			Input:      map[string]any{"command": "ls"},
		},
		ToolApprovalResponsePart{
			ApprovalID:       "apr_1",
			Approved:         true,
			Reason:           "ok",
			ProviderExecuted: true,
		},
	)

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"type":"tool-approval-request"`) {
		t.Errorf("missing tool-approval-request type: %s", data)
	}
	if !strings.Contains(string(data), `"type":"tool-approval-response"`) {
		t.Errorf("missing tool-approval-response type: %s", data)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Content.Parts))
	}
	req, ok := got.Content.Parts[0].(ToolApprovalRequestPart)
	if !ok {
		t.Fatalf("part[0] = %T, want ToolApprovalRequestPart", got.Content.Parts[0])
	}
	if req.ApprovalID != "apr_1" || req.ToolCallID != "call_1" || req.ToolName != "bash" {
		t.Errorf("approval request not preserved: %+v", req)
	}
	resp, ok := got.Content.Parts[1].(ToolApprovalResponsePart)
	if !ok {
		t.Fatalf("part[1] = %T, want ToolApprovalResponsePart", got.Content.Parts[1])
	}
	if resp.ApprovalID != "apr_1" || !resp.Approved || resp.Reason != "ok" || !resp.ProviderExecuted {
		t.Errorf("approval response not preserved: %+v", resp)
	}
}
