package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// FatalToolError is an interface that errors can implement to indicate
// they should not be fed back to the model as tool results. When a tool
// returns an error implementing this interface, the executor propagates it
// instead of converting it to a Response. This is used for errors like
// permission timeouts that should suspend the run, not be retried by the model.
type FatalToolError interface {
	error
	FatalToolError() bool
}

// DebugExecutor enables debug logging for tool execution
var DebugExecutor = os.Getenv("DEBUG_EXECUTOR") != ""

// Executor abstracts tool execution, allowing tools to run locally or remotely.
// This enables separating the LLM orchestration from tool execution,
// supporting architectures where tools run in isolated containers.
type Executor interface {
	// Execute runs a tool and returns the result.
	// The executor handles all local concerns (file system, permissions, etc.)
	Execute(ctx context.Context, req Request) (Response, error)

	// Tools returns information about available tools.
	// This is used to build the tool list for the LLM.
	Tools() []Info
}

// Request represents a tool execution request.
type Request struct {
	// ToolCallID is the unique identifier for this tool call (from LLM)
	ToolCallID string `json:"tool_call_id"`

	// ToolName is the name of the tool to execute
	ToolName string `json:"tool_name"`

	// Input is the tool-specific input (JSON)
	Input json.RawMessage `json:"input"`

	// SessionID identifies the session for file tracking and permissions
	SessionID string `json:"session_id,omitempty"`

	// WorkDir is the working directory for the tool
	WorkDir string `json:"work_dir,omitempty"`
}

// Response represents a tool execution response.
type Response struct {
	// Output is the tool's text output
	Output string `json:"output"`

	// Title is an optional title for the result
	Title string `json:"title,omitempty"`

	// Error is set if the tool execution failed
	Error string `json:"error,omitempty"`

	// IsError indicates whether this response represents an error
	IsError bool `json:"is_error,omitempty"`

	// Denied indicates the call was refused (permission/policy) rather than
	// failed. Distinct from IsError so it serializes to execution-denied.
	// DeniedReason is the optional human-facing reason.
	Denied       bool   `json:"denied,omitempty"`
	DeniedReason string `json:"denied_reason,omitempty"`

	// NoExecute is true when the tool has no execute function.
	// This matches ai-sdk behavior where tools without execute return undefined.
	NoExecute bool `json:"no_execute,omitempty"`

	// Metadata contains optional structured data
	Metadata map[string]any `json:"metadata,omitempty"`

	// Attachments contains optional file attachments
	Attachments []Attachment `json:"attachments,omitempty"`

	// PermissionRequired is set when the tool needs permission to proceed.
	// The orchestrator should handle this by requesting permission and
	// potentially re-executing the tool.
	PermissionRequired *PermissionRequest `json:"permission_required,omitempty"`
}

// PermissionRequest represents a request for permission to perform an action.
type PermissionRequest struct {
	// Permission type (e.g., "bash", "edit", "external_directory")
	Permission string `json:"permission"`

	// Patterns that need approval (e.g., command, file path)
	Patterns []string `json:"patterns"`

	// Description of what the tool wants to do
	Description string `json:"description,omitempty"`

	// Metadata contains additional context
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Info contains metadata about a tool for LLM consumption.
type Info struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// LocalExecutor executes tools in the current process.
// This is the default executor for CLI usage.
type LocalExecutor struct {
	tools       Set
	activeTools []string
}

// NewLocalExecutor creates a new local executor with the given tools.
func NewLocalExecutor(tools Set, activeTools []string) *LocalExecutor {
	return &LocalExecutor{
		tools:       tools,
		activeTools: activeTools,
	}
}

// Execute implements Executor.
func (e *LocalExecutor) Execute(ctx context.Context, req Request) (Response, error) {
	if DebugExecutor {
		fmt.Fprintf(os.Stderr, "[TOOL] >>> %s input=%s workdir=%s\n", req.ToolName, string(req.Input), req.WorkDir)
	}

	t, exists := e.tools[req.ToolName]
	if !exists {
		resp := Response{
			Output:  "Error: tool not found: " + req.ToolName,
			IsError: true,
		}
		if DebugExecutor {
			fmt.Fprintf(os.Stderr, "[TOOL] <<< %s error=not_found\n", req.ToolName)
		}
		return resp, nil
	}

	// Signal if no execute function (matches ai-sdk behavior where tools
	// without execute return undefined, which is then filtered out)
	if t.Execute == nil {
		if DebugExecutor {
			fmt.Fprintf(os.Stderr, "[TOOL] <<< %s skipped=no_execute_func\n", req.ToolName)
		}
		return Response{NoExecute: true}, nil
	}

	// Build context with session info
	if req.SessionID != "" {
		ctx = context.WithValue(ctx, SessionIDKey, req.SessionID)
	}
	if req.WorkDir != "" {
		ctx = context.WithValue(ctx, WorkDirKey, req.WorkDir)
	}

	// Execute the tool
	result, err := t.Execute(ctx, req.Input, CallOptions{
		ToolCallID:  req.ToolCallID,
		AbortSignal: ctx,
	})
	if err != nil {
		// Propagate context errors and fatal tool errors — these should not
		// be fed back to the model as tool results.
		if ctx.Err() != nil {
			return Response{}, ctx.Err()
		}
		if fatal, ok := err.(FatalToolError); ok && fatal.FatalToolError() {
			return Response{}, err
		}

		// A denied tool call is refused, not failed — surface it distinctly
		// so it serializes to execution-denied.
		var denied DeniedError
		if errors.As(err, &denied) {
			resp := Response{Denied: true, DeniedReason: denied.Reason}
			if DebugExecutor {
				fmt.Fprintf(os.Stderr, "[TOOL] <<< %s denied=%q\n", req.ToolName, denied.Reason)
			}
			return resp, nil
		}

		// Normal tool errors: convert to response for model feedback
		resp := Response{
			Output:  "Error: " + err.Error(),
			Error:   err.Error(),
			IsError: true,
		}
		if DebugExecutor {
			fmt.Fprintf(os.Stderr, "[TOOL] <<< %s error=%v\n", req.ToolName, err)
		}
		return resp, nil
	}

	if DebugExecutor {
		output := result.Output
		if len(output) > 200 {
			output = output[:200] + "..."
		}
		fmt.Fprintf(os.Stderr, "[TOOL] <<< %s output=%q\n", req.ToolName, output)
	}

	return Response{
		Output:      result.Output,
		Title:       result.Title,
		Metadata:    result.Metadata,
		Attachments: result.Attachments,
	}, nil
}

// Tools implements Executor.
func (e *LocalExecutor) Tools() []Info {
	ordered := e.tools.Ordered(e.activeTools)
	infos := make([]Info, len(ordered))
	for i, t := range ordered {
		infos[i] = Info{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return infos
}

// SetActiveTools updates the active tools filter.
func (e *LocalExecutor) SetActiveTools(activeTools []string) {
	e.activeTools = activeTools
}

// ContextKey is the type for tool execution context keys.
type ContextKey string

// Context keys for tool execution.
// These are used by LocalExecutor and should be used by tool implementations.
const (
	// SessionIDKey is the context key for session identification.
	SessionIDKey ContextKey = "sessionID"

	// WorkDirKey is the context key for the working directory.
	WorkDirKey ContextKey = "workDir"

	// RunnerKey is the context key for the runner instance (optional).
	RunnerKey ContextKey = "runner"
)
