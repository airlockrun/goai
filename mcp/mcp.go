// Package mcp adapts the official MCP Go SDK to GoAI tools.
package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/airlockrun/goai/tool"
)

type ServerConfig struct {
	Name         string
	ClientName   string
	Transport    string
	Command      string
	Args         []string
	Env          map[string]string
	URL          string
	Headers      map[string]string
	AuthProvider OAuthClientProvider
	HTTPClient   *http.Client
}

type Resource struct{ URI, Name, Description, MimeType string }
type ResourceContent struct{ URI, MimeType, Text, Blob string }

type Client struct {
	mu      sync.RWMutex
	servers map[string]*ServerConnection
}
type ServerConnection struct {
	mu        sync.Mutex
	config    ServerConfig
	session   *Session
	tools     tool.Set
	resources map[string]Resource
}

func NewClient() *Client { return &Client{servers: map[string]*ServerConnection{}} }

func (c *Client) Connect(ctx context.Context, config ServerConfig) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.servers[config.Name]; ok {
		return fmt.Errorf("server %q already connected", config.Name)
	}
	s, err := connect(ctx, config, true)
	if err != nil {
		return err
	}
	conn := &ServerConnection{config: config, session: s, tools: tool.Set{}, resources: map[string]Resource{}}
	if err := conn.refresh(ctx); err != nil {
		s.Close()
		return err
	}
	c.servers[config.Name] = conn
	return nil
}

func (conn *ServerConnection) refresh(ctx context.Context) error {
	s := conn.session
	conn.session.changed.Store(false)
	defer func() {
		if conn.tools == nil {
			conn.session.changed.Store(true)
		}
	}()
	conn.tools = nil
	tools := tool.Set{}
	resourceMap := map[string]Resource{}
	definitions, err := s.ListTools(ctx)
	if err != nil {
		return err
	}
	for _, definition := range definitions {
		input, err := json.Marshal(definition.InputSchema)
		if err != nil {
			return err
		}
		var output json.RawMessage
		if definition.OutputSchema != nil {
			output, err = json.Marshal(definition.OutputSchema)
			if err != nil {
				return err
			}
		}
		name := conn.config.Name + "_" + definition.Name
		tools[name] = tool.Tool{Name: name, Description: definition.Description, InputSchema: input, OutputSchema: output, Execute: conn.createToolExecutor(definition.Name)}
	}
	resources, err := s.ListResources(ctx)
	if err != nil {
		return err
	}
	for _, r := range resources {
		resourceMap[r.URI] = Resource{r.URI, r.Name, r.Description, r.MIMEType}
	}
	conn.tools, conn.resources = tools, resourceMap
	return nil
}

func (c *Client) Disconnect(name string) error {
	c.mu.Lock()
	conn, ok := c.servers[name]
	delete(c.servers, name)
	c.mu.Unlock()
	if !ok {
		return fmt.Errorf("server %q not connected", name)
	}
	return conn.session.Close()
}
func (c *Client) DisconnectAll() error {
	c.mu.Lock()
	servers := c.servers
	c.servers = map[string]*ServerConnection{}
	c.mu.Unlock()
	var err error
	for _, conn := range servers {
		err = errors.Join(err, conn.session.Close())
	}
	return err
}
func (c *Client) GetTools(ctx context.Context) (tool.Set, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := tool.Set{}
	for _, conn := range c.servers {
		conn.mu.Lock()
		if conn.session.changed.Load() {
			if err := conn.refresh(ctx); err != nil {
				conn.mu.Unlock()
				return nil, err
			}
		}
		for name, t := range conn.tools {
			if _, exists := out[name]; exists {
				conn.mu.Unlock()
				return nil, fmt.Errorf("MCP tool name collision: %q", name)
			}
			out[name] = t
		}
		conn.mu.Unlock()
	}
	return out, nil
}
func (c *Client) GetResources(ctx context.Context) ([]Resource, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []Resource
	for _, conn := range c.servers {
		conn.mu.Lock()
		if conn.session.changed.Load() {
			if err := conn.refresh(ctx); err != nil {
				conn.mu.Unlock()
				return nil, err
			}
		}
		for _, r := range conn.resources {
			out = append(out, r)
		}
		conn.mu.Unlock()
	}
	return out, nil
}
func (c *Client) GetServerInstructions(name string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if conn := c.servers[name]; conn != nil {
		return conn.session.Instructions()
	}
	return ""
}
func (c *Client) ReadResource(ctx context.Context, uri string) (*ResourceContent, error) {
	c.mu.RLock()
	var selected *ServerConnection
	for _, conn := range c.servers {
		conn.mu.Lock()
		if conn.session.changed.Load() {
			if err := conn.refresh(ctx); err != nil {
				conn.mu.Unlock()
				c.mu.RUnlock()
				return nil, err
			}
		}
		_, ok := conn.resources[uri]
		conn.mu.Unlock()
		if ok {
			if selected != nil {
				c.mu.RUnlock()
				return nil, errors.New("resource URI belongs to multiple servers")
			}
			selected = conn
		}
	}
	c.mu.RUnlock()
	if selected == nil {
		return nil, fmt.Errorf("resource %q not found", uri)
	}
	result, err := selected.session.ReadResource(ctx, uri)
	if err != nil {
		return nil, err
	}
	if result.NeedsInput() {
		return nil, errors.New("MCP resource requires additional input")
	}
	if len(result.Contents) != 1 {
		return nil, errors.New("ReadResource requires exactly one content item; use Session.ReadResource for multipart resources")
	}
	r := result.Contents[0]
	return &ResourceContent{URI: r.URI, MimeType: r.MIMEType, Text: r.Text, Blob: base64.StdEncoding.EncodeToString(r.Blob)}, nil
}

func (c *ServerConnection) createToolExecutor(name string) tool.ExecuteFunc {
	return func(ctx context.Context, input json.RawMessage, opts tool.CallOptions) (tool.Result, error) {
		c.mu.Lock()
		if c.session.changed.Load() {
			if err := c.refresh(ctx); err != nil {
				c.mu.Unlock()
				return tool.Result{}, err
			}
		}
		_, exists := c.tools[c.config.Name+"_"+name]
		c.mu.Unlock()
		if !exists {
			return tool.Result{}, fmt.Errorf("MCP tool %q is unavailable", name)
		}
		result, err := c.session.CallTool(ctx, name, input)
		if err != nil {
			return tool.Result{}, err
		}
		if result.NeedsInput() {
			return tool.Result{}, errors.New("MCP tool requires additional input; call is incomplete")
		}
		raw, err := json.Marshal(result)
		if err != nil {
			return tool.Result{}, err
		}
		return projectResult(raw)
	}
}

// projectResult is the model-facing projection. Session.CallTool retains the
// full typed result, including ordered content and structured output.
func projectResult(raw json.RawMessage) (tool.Result, error) {
	var result struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Data     string `json:"data"`
			MimeType string `json:"mimeType"`
			URI      string `json:"uri"`
			Name     string `json:"name"`
			Resource *struct {
				Text     string `json:"text"`
				Blob     string `json:"blob"`
				MimeType string `json:"mimeType"`
			} `json:"resource"`
		} `json:"content"`
		StructuredContent json.RawMessage `json:"structuredContent"`
		IsError           bool            `json:"isError"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return tool.Result{}, err
	}
	out := tool.Result{Metadata: map[string]any{"mcp": raw}}
	var text []string
	for _, part := range result.Content {
		switch part.Type {
		case "text":
			text = append(text, part.Text)
		case "image", "audio":
			out.Attachments = append(out.Attachments, tool.Attachment{Data: part.Data, MimeType: part.MimeType})
		case "resource_link":
			text = append(text, fmt.Sprintf("[Resource: %s — %s]", part.Name, part.URI))
		case "resource":
			if part.Resource != nil {
				if part.Resource.Text != "" {
					text = append(text, part.Resource.Text)
				}
				if part.Resource.Blob != "" {
					out.Attachments = append(out.Attachments, tool.Attachment{Data: part.Resource.Blob, MimeType: part.Resource.MimeType})
				}
			}
		}
	}
	if len(result.Content) == 0 && len(result.StructuredContent) > 0 {
		text = append(text, string(result.StructuredContent))
	}
	out.Output = strings.Join(text, "\n")
	if result.IsError {
		return out, fmt.Errorf("MCP tool error: %s", out.Output)
	}
	return out, nil
}
