package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/airlockrun/goai/message"
)

func TestOutputForError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantType message.ToolResultOutput
		check    func(t *testing.T, out message.ToolResultOutput)
	}{
		{
			name:     "DeniedError maps to ExecutionDeniedOutput with reason",
			err:      DeniedError{Reason: "policy refused"},
			wantType: message.ExecutionDeniedOutput{},
			check: func(t *testing.T, out message.ToolResultOutput) {
				d, ok := out.(message.ExecutionDeniedOutput)
				if !ok {
					t.Fatalf("got %T, want ExecutionDeniedOutput", out)
				}
				if d.Reason != "policy refused" {
					t.Errorf("Reason = %q, want %q", d.Reason, "policy refused")
				}
			},
		},
		{
			name:     "wrapped DeniedError still maps to ExecutionDeniedOutput",
			err:      fmt.Errorf("outer: %w", DeniedError{Reason: "nested"}),
			wantType: message.ExecutionDeniedOutput{},
			check: func(t *testing.T, out message.ToolResultOutput) {
				d, ok := out.(message.ExecutionDeniedOutput)
				if !ok {
					t.Fatalf("got %T, want ExecutionDeniedOutput", out)
				}
				if d.Reason != "nested" {
					t.Errorf("Reason = %q, want %q", d.Reason, "nested")
				}
			},
		},
		{
			name:     "plain error maps to ErrorTextOutput",
			err:      errors.New("boom"),
			wantType: message.ErrorTextOutput{},
			check: func(t *testing.T, out message.ToolResultOutput) {
				e, ok := out.(message.ErrorTextOutput)
				if !ok {
					t.Fatalf("got %T, want ErrorTextOutput", out)
				}
				if e.Value != "boom" {
					t.Errorf("Value = %q, want %q", e.Value, "boom")
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := OutputForError(tt.err)
			if reflect.TypeOf(out) != reflect.TypeOf(tt.wantType) {
				t.Fatalf("type = %T, want %T", out, tt.wantType)
			}
			tt.check(t, out)
		})
	}
}

func TestSuccessOutput(t *testing.T) {
	t.Run("no attachments yields TextOutput", func(t *testing.T) {
		out := SuccessOutput(Result{Output: "all good"})
		to, ok := out.(message.TextOutput)
		if !ok {
			t.Fatalf("got %T, want TextOutput", out)
		}
		if to.Value != "all good" {
			t.Errorf("Value = %q, want %q", to.Value, "all good")
		}
	})

	t.Run("image attachment yields ContentOutput with text then image-data", func(t *testing.T) {
		out := SuccessOutput(Result{
			Output: "see image",
			Attachments: []Attachment{
				{Data: "aW1n", MimeType: "image/png", Filename: "shot.png"},
			},
		})
		co, ok := out.(message.ContentOutput)
		if !ok {
			t.Fatalf("got %T, want ContentOutput", out)
		}
		if len(co.Value) != 2 {
			t.Fatalf("expected 2 items, got %d: %+v", len(co.Value), co.Value)
		}
		if co.Value[0].Type != "text" || co.Value[0].Text != "see image" {
			t.Errorf("item[0] = %+v, want text 'see image'", co.Value[0])
		}
		if co.Value[1].Type != "image-data" {
			t.Errorf("item[1].Type = %q, want image-data", co.Value[1].Type)
		}
		if co.Value[1].Data != "aW1n" || co.Value[1].MediaType != "image/png" {
			t.Errorf("item[1] = %+v", co.Value[1])
		}
	})

	t.Run("non-image attachment yields file-data item", func(t *testing.T) {
		out := SuccessOutput(Result{
			Output: "doc",
			Attachments: []Attachment{
				{Data: "cGRm", MimeType: "application/pdf", Filename: "report.pdf"},
			},
		})
		co, ok := out.(message.ContentOutput)
		if !ok {
			t.Fatalf("got %T, want ContentOutput", out)
		}
		if co.Value[1].Type != "file-data" || co.Value[1].Filename != "report.pdf" {
			t.Errorf("item[1] = %+v, want file-data report.pdf", co.Value[1])
		}
	})
}

// TestLocalExecutor_DeniedError verifies a tool whose Execute returns a
// DeniedError surfaces as Response{Denied:true, DeniedReason:...} (not an
// IsError response) — the end-to-end denied path (ADR §8 item 4).
func TestLocalExecutor_DeniedError(t *testing.T) {
	tools := Set{
		"guarded": Tool{
			Name: "guarded",
			Execute: func(ctx context.Context, input json.RawMessage, opts CallOptions) (Result, error) {
				return Result{}, DeniedError{Reason: "nope"}
			},
		},
	}
	exec := NewLocalExecutor(tools, []string{"guarded"})
	resp, err := exec.Execute(context.Background(), Request{ToolName: "guarded", ToolCallID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !resp.Denied {
		t.Errorf("Denied = false, want true (resp = %+v)", resp)
	}
	if resp.DeniedReason != "nope" {
		t.Errorf("DeniedReason = %q, want %q", resp.DeniedReason, "nope")
	}
	if resp.IsError {
		t.Error("denied response must not also be IsError")
	}

	// And the response maps to an ExecutionDeniedOutput via SuccessOutput's
	// sibling classification path: confirm OutputForError on the same error
	// produces the matching denied output.
	out := OutputForError(DeniedError{Reason: "nope"})
	if d, ok := out.(message.ExecutionDeniedOutput); !ok || d.Reason != "nope" {
		t.Errorf("OutputForError = %#v, want ExecutionDeniedOutput{Reason:\"nope\"}", out)
	}
}
