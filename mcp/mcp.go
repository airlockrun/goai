// Package mcp provides Model Context Protocol (MCP) client functionality.
// MCP enables connecting to external servers that provide tools and resources
// for AI models.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/airlockrun/goai/tool"
)

// Client manages connections to MCP servers.
type Client struct {
	servers map[string]*ServerConnection
	mu      sync.RWMutex
}

// NewClient creates a new MCP client.
func NewClient() *Client {
	return &Client{
		servers: make(map[string]*ServerConnection),
	}
}

// ServerConfig contains configuration for an MCP server.
type ServerConfig struct {
	// Name is a unique identifier for this server.
	Name string

	// Transport specifies how to connect to the server.
	// Options: "stdio", "sse", "http"
	Transport string

	// Command is the command to run for stdio transport.
	Command string

	// Args are arguments for the command.
	Args []string

	// Env are environment variables for the command.
	Env map[string]string

	// URL is the server URL for sse/http transport.
	URL string

	// Headers are HTTP headers for sse/http transport.
	Headers map[string]string

	// AuthProvider is the optional OAuth integration. nil = no OAuth.
	AuthProvider OAuthClientProvider
}

// ServerConnection represents a connection to an MCP server.
type ServerConnection struct {
	config    ServerConfig
	transport Transport
	tools     map[string]tool.Tool
	resources map[string]Resource
	mu        sync.RWMutex
	connected bool
}

// Transport is the interface for MCP transports.
type Transport interface {
	// Connect establishes the connection.
	Connect(ctx context.Context) error

	// Close closes the connection.
	Close() error

	// Send sends a request and returns the response.
	Send(ctx context.Context, method string, params any) (json.RawMessage, error)

	// OnNotification registers a handler for server notifications.
	OnNotification(handler func(method string, params json.RawMessage))
}

// Resource represents an MCP resource.
type Resource struct {
	// URI is the unique identifier for the resource.
	URI string

	// Name is the human-readable name.
	Name string

	// Description describes the resource.
	Description string

	// MimeType is the content type.
	MimeType string
}

// ResourceContent contains the content of a resource.
type ResourceContent struct {
	// URI is the resource URI.
	URI string

	// MimeType is the content type.
	MimeType string

	// Text is the text content (for text resources).
	Text string

	// Blob is the binary content (for binary resources, base64 encoded).
	Blob string
}

// Connect connects to an MCP server.
func (c *Client) Connect(ctx context.Context, config ServerConfig) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if already connected
	if _, exists := c.servers[config.Name]; exists {
		return fmt.Errorf("server %q already connected", config.Name)
	}

	// Create transport based on config
	var transport Transport
	switch config.Transport {
	case "stdio":
		transport = NewStdioTransport(config.Command, config.Args, config.Env)
	case "sse":
		transport = NewSSETransport(config.URL, config.Headers, config.AuthProvider)
	case "http":
		transport = NewHTTPTransport(config.URL, config.Headers, config.AuthProvider)
	default:
		return fmt.Errorf("unknown transport: %s", config.Transport)
	}

	// Connect
	if err := transport.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	conn := &ServerConnection{
		config:    config,
		transport: transport,
		tools:     make(map[string]tool.Tool),
		resources: make(map[string]Resource),
		connected: true,
	}
	transport.OnNotification(conn.handleNotification)

	// Initialize the connection
	if err := conn.initialize(ctx); err != nil {
		transport.Close()
		return fmt.Errorf("failed to initialize: %w", err)
	}

	c.servers[config.Name] = conn
	return nil
}

// Disconnect disconnects from an MCP server.
func (c *Client) Disconnect(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	conn, exists := c.servers[name]
	if !exists {
		return fmt.Errorf("server %q not connected", name)
	}

	conn.mu.Lock()
	conn.connected = false
	conn.mu.Unlock()

	if err := conn.transport.Close(); err != nil {
		return fmt.Errorf("failed to close connection: %w", err)
	}

	delete(c.servers, name)
	return nil
}

// DisconnectAll disconnects from all MCP servers.
func (c *Client) DisconnectAll() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	var lastErr error
	for name, conn := range c.servers {
		conn.mu.Lock()
		conn.connected = false
		conn.mu.Unlock()

		if err := conn.transport.Close(); err != nil {
			lastErr = err
		}
		delete(c.servers, name)
	}
	return lastErr
}

// GetTools returns all tools from all connected servers.
func (c *Client) GetTools() tool.Set {
	c.mu.RLock()
	defer c.mu.RUnlock()

	tools := make(tool.Set)
	for _, conn := range c.servers {
		conn.mu.RLock()
		for name, t := range conn.tools {
			tools[name] = t
		}
		conn.mu.RUnlock()
	}
	return tools
}

// GetResources returns all resources from all connected servers.
func (c *Client) GetResources() []Resource {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var resources []Resource
	for _, conn := range c.servers {
		conn.mu.RLock()
		for _, r := range conn.resources {
			resources = append(resources, r)
		}
		conn.mu.RUnlock()
	}
	return resources
}

// ReadResource reads the content of a resource.
func (c *Client) ReadResource(ctx context.Context, uri string) (*ResourceContent, error) {
	c.mu.RLock()

	// Find the server that owns this resource
	var conn *ServerConnection
	for _, s := range c.servers {
		s.mu.RLock()
		if _, exists := s.resources[uri]; exists {
			conn = s
			s.mu.RUnlock()
			break
		}
		s.mu.RUnlock()
	}
	c.mu.RUnlock()

	if conn == nil {
		return nil, fmt.Errorf("resource %q not found", uri)
	}

	return conn.readResource(ctx, uri)
}

// ServerConnection methods

func (conn *ServerConnection) initialize(ctx context.Context) error {
	// Send initialize request
	initParams := map[string]any{
		"protocolVersion": LatestProtocolVersion,
		"capabilities": map[string]any{
			"tools":     map[string]any{},
			"resources": map[string]any{"subscribe": true},
		},
		"clientInfo": map[string]any{
			"name":    "goai",
			"version": "1.0.0",
		},
	}

	_, err := conn.transport.Send(ctx, "initialize", initParams)
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}

	// Send initialized notification
	_, err = conn.transport.Send(ctx, "notifications/initialized", nil)
	if err != nil {
		return fmt.Errorf("initialized notification failed: %w", err)
	}

	// List tools
	if err := conn.listTools(ctx); err != nil {
		return fmt.Errorf("list tools failed: %w", err)
	}

	// List resources
	if err := conn.listResources(ctx); err != nil {
		return fmt.Errorf("list resources failed: %w", err)
	}

	return nil
}

// handleNotification dispatches server-pushed notifications. The well-
// known list_changed notifications trigger a re-fetch so our local view
// stays in sync; everything else is silently ignored for now.
func (conn *ServerConnection) handleNotification(method string, _ json.RawMessage) {
	ctx := context.Background()
	switch method {
	case "notifications/tools/list_changed":
		_ = conn.listTools(ctx)
	case "notifications/resources/list_changed":
		_ = conn.listResources(ctx)
	}
}

func (conn *ServerConnection) listTools(ctx context.Context) error {
	result, err := conn.transport.Send(ctx, "tools/list", nil)
	if err != nil {
		return err
	}

	var response struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"inputSchema"`
		} `json:"tools"`
	}

	if err := json.Unmarshal(result, &response); err != nil {
		return err
	}

	conn.mu.Lock()
	defer conn.mu.Unlock()

	for _, t := range response.Tools {
		toolName := fmt.Sprintf("%s_%s", conn.config.Name, t.Name)
		conn.tools[toolName] = tool.Tool{
			Name:        toolName,
			Description: t.Description,
			InputSchema: t.InputSchema,
			Execute:     conn.createToolExecutor(t.Name),
		}
	}

	return nil
}

func (conn *ServerConnection) createToolExecutor(toolName string) tool.ExecuteFunc {
	return func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
		conn.mu.RLock()
		if !conn.connected {
			conn.mu.RUnlock()
			return tool.Result{}, fmt.Errorf("server not connected")
		}
		conn.mu.RUnlock()

		var args map[string]any
		if err := json.Unmarshal(input, &args); err != nil {
			return tool.Result{}, err
		}

		params := map[string]any{
			"name":      toolName,
			"arguments": args,
		}

		result, err := conn.transport.Send(ctx, "tools/call", params)
		if err != nil {
			return tool.Result{}, err
		}

		var response struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		}

		if err := json.Unmarshal(result, &response); err != nil {
			return tool.Result{}, err
		}

		if response.IsError {
			if len(response.Content) > 0 {
				return tool.Result{}, fmt.Errorf("tool error: %s", response.Content[0].Text)
			}
			return tool.Result{}, fmt.Errorf("tool error")
		}

		output := ""
		for _, c := range response.Content {
			if c.Type == "text" {
				output += c.Text
			}
		}

		return tool.Result{Output: output}, nil
	}
}

func (conn *ServerConnection) listResources(ctx context.Context) error {
	result, err := conn.transport.Send(ctx, "resources/list", nil)
	if err != nil {
		// Resources might not be supported
		return nil
	}

	var response struct {
		Resources []struct {
			URI         string `json:"uri"`
			Name        string `json:"name"`
			Description string `json:"description"`
			MimeType    string `json:"mimeType"`
		} `json:"resources"`
	}

	if err := json.Unmarshal(result, &response); err != nil {
		return nil
	}

	conn.mu.Lock()
	defer conn.mu.Unlock()

	for _, r := range response.Resources {
		conn.resources[r.URI] = Resource{
			URI:         r.URI,
			Name:        r.Name,
			Description: r.Description,
			MimeType:    r.MimeType,
		}
	}

	return nil
}

func (conn *ServerConnection) readResource(ctx context.Context, uri string) (*ResourceContent, error) {
	conn.mu.RLock()
	if !conn.connected {
		conn.mu.RUnlock()
		return nil, fmt.Errorf("server not connected")
	}
	conn.mu.RUnlock()

	params := map[string]any{
		"uri": uri,
	}

	result, err := conn.transport.Send(ctx, "resources/read", params)
	if err != nil {
		return nil, err
	}

	var response struct {
		Contents []struct {
			URI      string `json:"uri"`
			MimeType string `json:"mimeType"`
			Text     string `json:"text,omitempty"`
			Blob     string `json:"blob,omitempty"`
		} `json:"contents"`
	}

	if err := json.Unmarshal(result, &response); err != nil {
		return nil, err
	}

	if len(response.Contents) == 0 {
		return nil, fmt.Errorf("no content returned")
	}

	c := response.Contents[0]
	return &ResourceContent{
		URI:      c.URI,
		MimeType: c.MimeType,
		Text:     c.Text,
		Blob:     c.Blob,
	}, nil
}
