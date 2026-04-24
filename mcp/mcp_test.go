package mcp

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/airlockrun/goai/tool"
)

// connectWithMock creates a Client connected to a MockTransport with the given server name.
// It bypasses the normal Connect flow to inject the mock transport directly.
func connectWithMock(t *testing.T, name string, mock *MockTransport) *Client {
	t.Helper()
	client := NewClient()
	conn := &ServerConnection{
		config:    ServerConfig{Name: name},
		transport: mock,
		tools:     make(map[string]tool.Tool),
		resources: make(map[string]Resource),
		connected: true,
	}
	if err := mock.Connect(context.Background()); err != nil {
		t.Fatalf("mock connect failed: %v", err)
	}
	if err := conn.initialize(context.Background()); err != nil {
		t.Fatalf("initialize failed: %v", err)
	}
	client.servers[name] = conn
	return client
}

func TestClientToolDiscovery(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()

	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	if _, ok := tools["test_mock-tool"]; !ok {
		t.Error("missing tool test_mock-tool")
	}
	if _, ok := tools["test_mock-tool-no-args"]; !ok {
		t.Error("missing tool test_mock-tool-no-args")
	}

	// Check tool has correct description
	tt := tools["test_mock-tool"]
	if tt.Description != "A mock tool for testing" {
		t.Errorf("unexpected description: %s", tt.Description)
	}

	// Check input schema is preserved
	var schema map[string]any
	if err := json.Unmarshal(tt.InputSchema, &schema); err != nil {
		t.Fatalf("failed to parse schema: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("expected type=object, got %v", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("missing properties in schema")
	}
	if _, ok := props["foo"]; !ok {
		t.Error("missing property 'foo' in schema")
	}
}

func TestClientToolExecution(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	tt := tools["test_mock-tool"]

	result, err := tt.Execute(context.Background(), json.RawMessage(`{"foo":"bar"}`), tool.CallOptions{})
	if err != nil {
		t.Fatalf("tool execution failed: %v", err)
	}

	if result.Output != "Mock tool call result" {
		t.Errorf("unexpected output: %s", result.Output)
	}
}

func TestClientToolExecutionCustomResult(t *testing.T) {
	mock := NewMockTransport()
	mock.ToolCallResults["mock-tool"] = MockToolResult{
		Content: []MockContent{{Type: "text", Text: "Custom result for mock-tool"}},
	}

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	result, err := tools["test_mock-tool"].Execute(context.Background(), json.RawMessage(`{"foo":"bar"}`), tool.CallOptions{})
	if err != nil {
		t.Fatalf("tool execution failed: %v", err)
	}

	if result.Output != "Custom result for mock-tool" {
		t.Errorf("unexpected output: %s", result.Output)
	}
}

func TestClientZeroArgTool(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	tt := tools["test_mock-tool-no-args"]

	if tt.InputSchema == nil {
		t.Fatal("expected input schema to be present")
	}

	result, err := tt.Execute(context.Background(), json.RawMessage(`{}`), tool.CallOptions{})
	if err != nil {
		t.Fatalf("tool execution failed: %v", err)
	}

	if result.Output != "Mock tool call result" {
		t.Errorf("unexpected output: %s", result.Output)
	}
}

func TestClientToolNotFound(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	// Call a non-existent tool via executor
	conn := client.servers["test"]
	executor := conn.createToolExecutor("nonexistent-tool")
	_, err := executor(context.Background(), json.RawMessage(`{}`), tool.CallOptions{})
	if err == nil {
		t.Fatal("expected error for nonexistent tool")
	}

	// The real tool should work fine
	tools := client.GetTools()
	_, err = tools["test_mock-tool"].Execute(context.Background(), json.RawMessage(`{"foo":"bar"}`), tool.CallOptions{})
	if err != nil {
		t.Fatalf("valid tool call failed: %v", err)
	}
}

func TestClientToolInvalidParams(t *testing.T) {
	mock := NewMockTransport()
	mock.FailOnInvalidToolParams = true

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	_, err := tools["test_mock-tool"].Execute(context.Background(), json.RawMessage(`{"bar":"baz"}`), tool.CallOptions{})
	if err == nil {
		t.Fatal("expected error for invalid tool params")
	}
}

func TestClientResourceList(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	resources := client.GetResources()
	if len(resources) != 1 {
		t.Fatalf("expected 1 resource, got %d", len(resources))
	}

	r := resources[0]
	if r.URI != "file:///mock/resource.txt" {
		t.Errorf("unexpected URI: %s", r.URI)
	}
	if r.Name != "resource.txt" {
		t.Errorf("unexpected name: %s", r.Name)
	}
	if r.Description != "Mock resource" {
		t.Errorf("unexpected description: %s", r.Description)
	}
	if r.MimeType != "text/plain" {
		t.Errorf("unexpected mime type: %s", r.MimeType)
	}
}

func TestClientResourceRead(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	content, err := client.ReadResource(context.Background(), "file:///mock/resource.txt")
	if err != nil {
		t.Fatalf("ReadResource failed: %v", err)
	}

	if content.Text != "Mock resource content" {
		t.Errorf("unexpected text: %s", content.Text)
	}
	if content.MimeType != "text/plain" {
		t.Errorf("unexpected mime type: %s", content.MimeType)
	}
}

func TestClientResourceNotFound(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	_, err := client.ReadResource(context.Background(), "file:///nonexistent.txt")
	if err == nil {
		t.Fatal("expected error for nonexistent resource")
	}
}

func TestClientConnectError(t *testing.T) {
	mock := NewMockTransport()
	mock.SendError = true

	err := mock.Connect(context.Background())
	if err == nil {
		t.Fatal("expected connect error")
	}
}

func TestClientDisconnect(t *testing.T) {
	mock := NewMockTransport()
	client := connectWithMock(t, "test", mock)

	if err := client.Disconnect("test"); err != nil {
		t.Fatalf("disconnect failed: %v", err)
	}

	if !mock.closed {
		t.Error("expected transport to be closed")
	}

	tools := client.GetTools()
	if len(tools) != 0 {
		t.Errorf("expected 0 tools after disconnect, got %d", len(tools))
	}
}

func TestClientDisconnectAll(t *testing.T) {
	mock1 := NewMockTransport()
	mock2 := NewMockTransport()

	client := connectWithMock(t, "server1", mock1)
	// Add second server
	conn2 := &ServerConnection{
		config:    ServerConfig{Name: "server2"},
		transport: mock2,
		tools:     make(map[string]tool.Tool),
		resources: make(map[string]Resource),
	}
	mock2.Connect(context.Background())
	conn2.initialize(context.Background())
	client.servers["server2"] = conn2

	if err := client.DisconnectAll(); err != nil {
		t.Fatalf("disconnect all failed: %v", err)
	}

	if !mock1.closed || !mock2.closed {
		t.Error("expected both transports to be closed")
	}
}

func TestClientMultipleServers(t *testing.T) {
	mock1 := NewMockTransport()
	mock1.Tools = []MockTool{
		{Name: "tool-a", Description: "Tool A", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)},
	}
	mock1.Resources = []MockResource{}

	mock2 := NewMockTransport()
	mock2.Tools = []MockTool{
		{Name: "tool-b", Description: "Tool B", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)},
	}
	mock2.Resources = []MockResource{}

	client := connectWithMock(t, "server1", mock1)
	conn2 := &ServerConnection{
		config:    ServerConfig{Name: "server2"},
		transport: mock2,
		tools:     make(map[string]tool.Tool),
		resources: make(map[string]Resource),
	}
	mock2.Connect(context.Background())
	conn2.initialize(context.Background())
	client.servers["server2"] = conn2

	defer client.DisconnectAll()

	tools := client.GetTools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	if _, ok := tools["server1_tool-a"]; !ok {
		t.Error("missing server1_tool-a")
	}
	if _, ok := tools["server2_tool-b"]; !ok {
		t.Error("missing server2_tool-b")
	}
}

func TestClientCustomTools(t *testing.T) {
	mock := NewMockTransport()
	mock.Tools = []MockTool{
		{
			Name:        "custom-tool",
			Description: "A custom tool",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}}}`),
		},
	}
	mock.Resources = []MockResource{}
	mock.ToolCallResults["custom-tool"] = MockToolResult{
		Content: []MockContent{{Type: "text", Text: "custom result"}},
	}

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	result, err := tools["test_custom-tool"].Execute(context.Background(), json.RawMessage(`{"input":"hello"}`), tool.CallOptions{})
	if err != nil {
		t.Fatalf("execution failed: %v", err)
	}
	if result.Output != "custom result" {
		t.Errorf("unexpected output: %s", result.Output)
	}
}

func TestClientNoTools(t *testing.T) {
	mock := NewMockTransport()
	mock.Tools = []MockTool{}
	mock.Resources = []MockResource{}

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	if len(tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(tools))
	}
}

func TestClientNoResources(t *testing.T) {
	mock := NewMockTransport()
	mock.Resources = []MockResource{}
	mock.ResourceContents = []MockResourceContent{}

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	resources := client.GetResources()
	if len(resources) != 0 {
		t.Errorf("expected 0 resources, got %d", len(resources))
	}
}

func TestClientToolErrorResult(t *testing.T) {
	mock := NewMockTransport()
	mock.ToolCallResults["mock-tool"] = MockToolResult{
		Content: []MockContent{{Type: "text", Text: "something went wrong"}},
		IsError: true,
	}

	client := connectWithMock(t, "test", mock)
	defer client.DisconnectAll()

	tools := client.GetTools()
	_, err := tools["test_mock-tool"].Execute(context.Background(), json.RawMessage(`{"foo":"bar"}`), tool.CallOptions{})
	if err == nil {
		t.Fatal("expected error for isError=true result")
	}
}
