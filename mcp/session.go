package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"sync"
	"sync/atomic"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const MaxHTTPResponseBytes = 20 << 20

// Session owns an official MCP SDK connection. Callers must close it.
// Protocol-aware callers retain the SDK's complete typed results.
type Session struct {
	client    *sdk.ClientSession
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once
	closeErr  error
	changed   atomic.Bool
}

// Connect opens a connection without projecting its results into model output.
func Connect(ctx context.Context, config ServerConfig) (*Session, error) {
	return connect(ctx, config, false)
}

func connect(ctx context.Context, config ServerConfig, watch bool) (*Session, error) {
	if config.HTTPClient == nil && config.Transport != "stdio" {
		panic("mcp: HTTP client is required")
	}
	life, cancel := context.WithCancel(context.Background())
	s := &Session{ctx: life, cancel: cancel}
	name := config.ClientName
	if name == "" {
		name = "goai"
	}
	opts := &sdk.ClientOptions{
		Capabilities:   &sdk.ClientCapabilities{},
		MultiRoundTrip: &sdk.MultiRoundTripOptions{Disabled: true},
	}
	if watch {
		opts.ToolListChangedHandler = func(context.Context, *sdk.ToolListChangedRequest) { s.changed.Store(true) }
		opts.ResourceListChangedHandler = func(context.Context, *sdk.ResourceListChangedRequest) { s.changed.Store(true) }
	}
	c := sdk.NewClient(&sdk.Implementation{Name: name, Version: "1.0.0"}, opts)
	var transport sdk.Transport
	switch config.Transport {
	case "http", "sse":
		client, err := sessionHTTPClient(config)
		if err != nil {
			cancel()
			return nil, err
		}
		if config.Transport == "http" {
			transport = &sdk.StreamableClientTransport{Endpoint: config.URL, HTTPClient: client, DisableStandaloneSSE: !watch, MaxEventSize: MaxHTTPResponseBytes}
		} else {
			transport = &sdk.SSEClientTransport{Endpoint: config.URL, HTTPClient: client, MaxEventSize: MaxHTTPResponseBytes}
		}
	case "stdio":
		cmd := exec.CommandContext(life, config.Command, config.Args...)
		cmd.Env = os.Environ()
		keys := make([]string, 0, len(config.Env))
		for key := range config.Env {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			cmd.Env = append(cmd.Env, key+"="+config.Env[key])
		}
		transport = &sdk.CommandTransport{Command: cmd}
	default:
		cancel()
		return nil, fmt.Errorf("unknown MCP transport %q", config.Transport)
	}
	tracked := &trackedTransport{Transport: transport}
	client, err := c.Connect(ctx, tracked, nil)
	if err != nil {
		cancel()
		if tracked.connection != nil {
			_ = tracked.connection.Close()
		}
		return nil, err
	}
	s.client = client
	return s, nil
}

func (s *Session) callContext(ctx context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(ctx)
	if s.ctx.Err() != nil {
		cancel()
	}
	stop := context.AfterFunc(s.ctx, cancel)
	return ctx, func() { stop(); cancel() }
}

// Keep the concrete SDK connection intact so negotiated transport state and
// session cleanup remain owned by the SDK, including failed discovery.
type trackedTransport struct {
	sdk.Transport
	connection sdk.Connection
}

func (t *trackedTransport) Connect(ctx context.Context) (sdk.Connection, error) {
	connection, err := t.Transport.Connect(ctx)
	t.connection = connection
	return connection, err
}

func (s *Session) Close() error {
	s.closeOnce.Do(func() { s.cancel(); s.closeErr = s.client.Close() })
	return s.closeErr
}

func (s *Session) Instructions() string { return s.client.InitializeResult().Instructions }

// ListTools returns a bounded complete discovery snapshot.
func (s *Session) ListTools(ctx context.Context) ([]*sdk.Tool, error) {
	ctx, cancel := s.callContext(ctx)
	defer cancel()
	var tools []*sdk.Tool
	seen := map[string]bool{}
	err := pages(ctx, func(cursor string) (string, any, error) {
		page, err := s.client.ListTools(ctx, &sdk.ListToolsParams{Cursor: cursor})
		if err != nil {
			return "", nil, err
		}
		for _, item := range page.Tools {
			if item == nil || item.Name == "" {
				return "", nil, errors.New("invalid MCP tool definition")
			}
			if seen[item.Name] {
				return "", nil, fmt.Errorf("duplicate tool name %q", item.Name)
			}
			seen[item.Name] = true
			tools = append(tools, item)
		}
		return page.NextCursor, page, nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	return tools, nil
}

func pages(ctx context.Context, next func(string) (string, any, error)) error {
	cursor, total := "", 0
	seen := map[string]bool{}
	for i := 0; i < 128; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		following, value, err := next(cursor)
		if err != nil {
			return err
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		total += len(encoded)
		if total > MaxHTTPResponseBytes {
			return errors.New("MCP discovery exceeds byte limit")
		}
		if following == "" {
			return nil
		}
		if seen[following] {
			return errors.New("MCP discovery repeated pagination cursor")
		}
		seen[following] = true
		cursor = following
	}
	return errors.New("MCP discovery exceeds page limit")
}

func (s *Session) ListResources(ctx context.Context) ([]*sdk.Resource, error) {
	ctx, cancel := s.callContext(ctx)
	defer cancel()
	caps := s.client.InitializeResult().Capabilities
	if caps == nil || caps.Resources == nil {
		return nil, nil
	}
	var resources []*sdk.Resource
	seen := map[string]bool{}
	err := pages(ctx, func(cursor string) (string, any, error) {
		page, err := s.client.ListResources(ctx, &sdk.ListResourcesParams{Cursor: cursor})
		if err != nil {
			return "", nil, err
		}
		for _, resource := range page.Resources {
			if resource == nil || resource.URI == "" {
				return "", nil, errors.New("invalid MCP resource definition")
			}
			if seen[resource.URI] {
				return "", nil, fmt.Errorf("duplicate MCP resource URI %q", resource.URI)
			}
			seen[resource.URI] = true
			resources = append(resources, resource)
		}
		return page.NextCursor, page, nil
	})
	return resources, err
}

func (s *Session) CallTool(ctx context.Context, name string, arguments json.RawMessage) (*sdk.CallToolResult, error) {
	ctx, cancel := s.callContext(ctx)
	defer cancel()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(arguments, &object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, errors.New("MCP arguments must be a JSON object")
	}
	return s.client.CallTool(ctx, &sdk.CallToolParams{Name: name, Arguments: arguments})
}

func (s *Session) ReadResource(ctx context.Context, uri string) (*sdk.ReadResourceResult, error) {
	ctx, cancel := s.callContext(ctx)
	defer cancel()
	return s.client.ReadResource(ctx, &sdk.ReadResourceParams{URI: uri})
}
