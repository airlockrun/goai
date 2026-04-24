// Package model defines the interfaces for different AI model types.
package model

import (
	"github.com/airlockrun/goai/stream"
)

// LanguageModel is the interface for text generation models.
// This is an alias for stream.Model for consistency with the existing codebase.
type LanguageModel = stream.Model

// LanguageModelInfo contains metadata about a language model.
type LanguageModelInfo struct {
	// ID is the model identifier.
	ID string

	// Provider is the provider identifier.
	Provider string

	// MaxTokens is the maximum number of tokens the model can generate.
	MaxTokens int

	// ContextWindow is the maximum context window size.
	ContextWindow int

	// SupportsTools indicates if the model supports tool/function calling.
	SupportsTools bool

	// SupportsVision indicates if the model supports image inputs.
	SupportsVision bool

	// SupportsStreaming indicates if the model supports streaming responses.
	SupportsStreaming bool

	// SupportsReasoning indicates if the model supports extended thinking.
	SupportsReasoning bool
}
