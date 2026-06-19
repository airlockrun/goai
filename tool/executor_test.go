package tool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestLocalExecutor_RecoversPanic verifies a panicking tool becomes a normal
// IsError response (carrying the stack) instead of crashing the process.
func TestLocalExecutor_RecoversPanic(t *testing.T) {
	tools := Set{}
	tools.Add(New("boom").
		Execute(func(ctx context.Context, input json.RawMessage, opts CallOptions) (Result, error) {
			panic("kaboom")
		}).
		Build())
	exec := NewLocalExecutor(tools, nil)

	resp, err := exec.Execute(context.Background(), Request{ToolName: "boom", Input: json.RawMessage(`{}`)})
	if err != nil {
		t.Fatalf("Execute returned err = %v; want nil (panic should become a response)", err)
	}
	if !resp.IsError {
		t.Fatalf("resp.IsError = false; want true")
	}
	if !strings.Contains(resp.Output, "boom panicked") {
		t.Errorf("resp.Output = %q; want it to mention the tool panicked", resp.Output)
	}
	if !strings.Contains(resp.Output, "kaboom") {
		t.Errorf("resp.Output = %q; want it to include the panic value", resp.Output)
	}
}

// TestLocalExecutor_SuccessUnaffected confirms the recover wrapper doesn't
// disturb the normal success path.
func TestLocalExecutor_SuccessUnaffected(t *testing.T) {
	tools := Set{}
	tools.Add(New("ok").
		Execute(func(ctx context.Context, input json.RawMessage, opts CallOptions) (Result, error) {
			return Result{Output: "done", Title: "ok-title"}, nil
		}).
		Build())
	exec := NewLocalExecutor(tools, nil)

	resp, err := exec.Execute(context.Background(), Request{ToolName: "ok", Input: json.RawMessage(`{}`)})
	if err != nil {
		t.Fatalf("Execute err = %v", err)
	}
	if resp.IsError {
		t.Fatalf("resp.IsError = true; want false")
	}
	if resp.Output != "done" || resp.Title != "ok-title" {
		t.Errorf("resp = %+v; want Output=done Title=ok-title", resp)
	}
}
