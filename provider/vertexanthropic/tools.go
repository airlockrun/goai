package vertexanthropic

import (
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/tool"
)

// Tools returns the Anthropic hosted-tool subset supported by Google Vertex.
// Mirrors ai-sdk's vertexAnthropicTools export at
// packages/google-vertex/src/anthropic/google-vertex-anthropic-provider.ts.
func (p *Provider) Tools() map[string]tool.Tool {
	return map[string]tool.Tool{
		"bash_20241022":            anthropic.Bash20241022(),
		"bash_20250124":            anthropic.Bash20250124(),
		"textEditor_20241022":      anthropic.TextEditor20241022(),
		"textEditor_20250124":      anthropic.TextEditor20250124(),
		"textEditor_20250429":      anthropic.TextEditor20250429(),
		"textEditor_20250728":      anthropic.TextEditor20250728(),
		"computer_20241022":        anthropic.Computer20241022With(anthropic.ComputerOptions{}),
		"webSearch_20250305":       anthropic.WebSearch20250305(),
		"toolSearchRegex_20251119": anthropic.ToolSearchRegex(),
		"toolSearchBm25_20251119":  anthropic.ToolSearchBM25(),
	}
}
