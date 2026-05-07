package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

// MockTransport implements the Transport interface for testing.
// It simulates an MCP server in-memory without any network I/O.
// Mirrors ai-sdk: packages/mcp/src/tool/mock-mcp-transport.ts
type MockTransport struct {
	Tools            []MockTool
	Resources        []MockResource
	ResourceContents []MockResourceContent
	ToolCallResults  map[string]MockToolResult // tool name -> custom result

	// FailOnInvalidToolParams makes tools/call return an error for any call.
	FailOnInvalidToolParams bool

	// InitializeResult overrides the initialize response.
	InitializeResult json.RawMessage

	// SendError makes Connect return an error.
	SendError bool

	closed        bool
	notifyHandler func(method string, params json.RawMessage)
}

// MockTool defines a tool in the mock server.
type MockTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// MockResource defines a resource in the mock server.
type MockResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// MockResourceContent defines the content returned for a resource.
type MockResourceContent struct {
	URI      string `json:"uri"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

// MockToolResult defines a custom tool call result.
type MockToolResult struct {
	Content []MockContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// MockContent is a content block in a tool result.
type MockContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// DefaultMockTools returns the default tools used by MockTransport.
func DefaultMockTools() []MockTool {
	return []MockTool{
		{
			Name:        "mock-tool",
			Description: "A mock tool for testing",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"foo":{"type":"string"}}}`),
		},
		{
			Name:        "mock-tool-no-args",
			Description: "A mock tool for testing",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
	}
}

// DefaultMockResources returns the default resources used by MockTransport.
func DefaultMockResources() []MockResource {
	return []MockResource{
		{
			URI:         "file:///mock/resource.txt",
			Name:        "resource.txt",
			Description: "Mock resource",
			MimeType:    "text/plain",
		},
	}
}

// DefaultMockResourceContents returns the default resource contents.
func DefaultMockResourceContents() []MockResourceContent {
	return []MockResourceContent{
		{
			URI:      "file:///mock/resource.txt",
			Text:     "Mock resource content",
			MimeType: "text/plain",
		},
	}
}

// NewMockTransport creates a MockTransport with default tools and resources.
func NewMockTransport() *MockTransport {
	return &MockTransport{
		Tools:            DefaultMockTools(),
		Resources:        DefaultMockResources(),
		ResourceContents: DefaultMockResourceContents(),
		ToolCallResults:  make(map[string]MockToolResult),
	}
}

func (t *MockTransport) Connect(ctx context.Context) error {
	if t.SendError {
		return fmt.Errorf("mock transport error")
	}
	return nil
}

func (t *MockTransport) Close() error {
	t.closed = true
	return nil
}

// SetProtocolVersion is a no-op on the mock — the mock doesn't enforce
// MCP-Protocol-Version headers. Implements Transport.
func (t *MockTransport) SetProtocolVersion(string) {}

func (t *MockTransport) OnNotification(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}

func (t *MockTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	if t.closed {
		return nil, fmt.Errorf("transport closed")
	}

	switch method {
	case "initialize":
		return t.handleInitialize()
	case "notifications/initialized":
		return nil, nil
	case "tools/list":
		return t.handleToolsList()
	case "tools/call":
		return t.handleToolsCall(params)
	case "resources/list":
		return t.handleResourcesList()
	case "resources/read":
		return t.handleResourcesRead(params)
	default:
		return nil, fmt.Errorf("unknown method: %s", method)
	}
}

func (t *MockTransport) handleInitialize() (json.RawMessage, error) {
	if t.InitializeResult != nil {
		return t.InitializeResult, nil
	}

	caps := map[string]any{}
	if len(t.Tools) > 0 {
		caps["tools"] = map[string]any{}
	}
	if len(t.Resources) > 0 {
		caps["resources"] = map[string]any{}
	}

	result := map[string]any{
		"protocolVersion": "2024-11-05",
		"serverInfo": map[string]any{
			"name":    "mock-mcp-server",
			"version": "1.0.0",
		},
		"capabilities": caps,
	}

	return json.Marshal(result)
}

func (t *MockTransport) handleToolsList() (json.RawMessage, error) {
	result := map[string]any{
		"tools": t.Tools,
	}
	return json.Marshal(result)
}

func (t *MockTransport) handleToolsCall(params any) (json.RawMessage, error) {
	// Extract tool name from params
	paramsBytes, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	var callParams struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(paramsBytes, &callParams); err != nil {
		return nil, err
	}

	// Check tool exists
	var found *MockTool
	for i := range t.Tools {
		if t.Tools[i].Name == callParams.Name {
			found = &t.Tools[i]
			break
		}
	}
	if found == nil {
		return nil, fmt.Errorf("RPC error -32601: Tool %s not found", callParams.Name)
	}

	if t.FailOnInvalidToolParams {
		return nil, fmt.Errorf("RPC error -32602: Invalid tool inputSchema: %v", callParams.Arguments)
	}

	// Check for custom result
	if custom, ok := t.ToolCallResults[callParams.Name]; ok {
		return json.Marshal(custom)
	}

	// Default result
	result := MockToolResult{
		Content: []MockContent{{Type: "text", Text: "Mock tool call result"}},
	}
	return json.Marshal(result)
}

func (t *MockTransport) handleResourcesList() (json.RawMessage, error) {
	result := map[string]any{
		"resources": t.Resources,
	}
	return json.Marshal(result)
}

func (t *MockTransport) handleResourcesRead(params any) (json.RawMessage, error) {
	paramsBytes, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	var readParams struct {
		URI string `json:"uri"`
	}
	if err := json.Unmarshal(paramsBytes, &readParams); err != nil {
		return nil, err
	}

	var contents []MockResourceContent
	for _, c := range t.ResourceContents {
		if c.URI == readParams.URI {
			contents = append(contents, c)
		}
	}

	if len(contents) == 0 {
		return nil, fmt.Errorf("RPC error -32002: Resource %s not found", readParams.URI)
	}

	result := map[string]any{
		"contents": contents,
	}
	return json.Marshal(result)
}
