package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

type executorFatalError struct {
	fatal bool
}

func (e executorFatalError) Error() string        { return "marked tool error" }
func (e executorFatalError) FatalToolError() bool { return e.fatal }

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

func TestLocalExecutor_WrappedFatalError(t *testing.T) {
	tests := []struct {
		name        string
		fatal       bool
		wantErr     bool
		wantIsError bool
	}{
		{name: "true marker propagates", fatal: true, wantErr: true},
		{name: "false marker becomes response", fatal: false, wantIsError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tools := Set{}
			tools.Add(New("marked").
				Execute(func(context.Context, json.RawMessage, CallOptions) (Result, error) {
					return Result{}, fmt.Errorf("wrapped: %w", executorFatalError{fatal: tt.fatal})
				}).
				Build())
			executor := NewLocalExecutor(tools, nil)

			resp, err := executor.Execute(context.Background(), Request{ToolName: "marked", Input: json.RawMessage(`{}`)})
			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}
			if resp.IsError != tt.wantIsError {
				t.Errorf("resp.IsError = %v, want %v", resp.IsError, tt.wantIsError)
			}
		})
	}
}
