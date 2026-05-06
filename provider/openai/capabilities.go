package openai

import "strings"

// LanguageModelCapabilities describes the capabilities of an OpenAI language model.
// Source: ai-sdk/packages/openai/src/openai-language-model-capabilities.ts
type LanguageModelCapabilities struct {
	// IsReasoningModel indicates if the model is a reasoning model (o1, o3, o4-mini, gpt-5, etc.)
	IsReasoningModel bool

	// SystemMessageMode determines how system messages should be handled.
	// "system" - use standard system role
	// "developer" - convert to developer role (for reasoning models)
	// "remove" - remove system messages entirely
	SystemMessageMode string

	// SupportsFlexProcessing indicates if the model supports flex processing tier.
	SupportsFlexProcessing bool

	// SupportsPriorityProcessing indicates if the model supports priority processing tier.
	SupportsPriorityProcessing bool

	// SupportsNonReasoningParameters indicates if the model allows temperature, topP, logProbs
	// when reasoningEffort is none. Only true for gpt-5.1+ models.
	SupportsNonReasoningParameters bool
}

// GetLanguageModelCapabilities returns the capabilities for a given OpenAI model ID.
func GetLanguageModelCapabilities(modelID string) LanguageModelCapabilities {
	supportsFlexProcessing := strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini") ||
		(strings.HasPrefix(modelID, "gpt-5") && !strings.HasPrefix(modelID, "gpt-5-chat"))

	supportsPriorityProcessing := strings.HasPrefix(modelID, "gpt-4") ||
		strings.HasPrefix(modelID, "gpt-5-mini") ||
		(strings.HasPrefix(modelID, "gpt-5") &&
			!strings.HasPrefix(modelID, "gpt-5-nano") &&
			!strings.HasPrefix(modelID, "gpt-5-chat")) ||
		strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini")

	// Use allowlist approach: only known reasoning models should use 'developer' role
	// This prevents issues with fine-tuned models, third-party models, and custom models
	isReasoningModel := strings.HasPrefix(modelID, "o1") ||
		strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini") ||
		strings.HasPrefix(modelID, "codex-mini") ||
		strings.HasPrefix(modelID, "computer-use-preview") ||
		(strings.HasPrefix(modelID, "gpt-5") && !strings.HasPrefix(modelID, "gpt-5-chat"))

	// https://platform.openai.com/docs/guides/latest-model#gpt-5-1-parameter-compatibility
	// GPT-5.1 and later families support temperature, topP, logProbs when
	// reasoningEffort is none.
	supportsNonReasoningParameters := strings.HasPrefix(modelID, "gpt-5.1") ||
		strings.HasPrefix(modelID, "gpt-5.2") ||
		strings.HasPrefix(modelID, "gpt-5.3") ||
		strings.HasPrefix(modelID, "gpt-5.4") ||
		strings.HasPrefix(modelID, "gpt-5.5")

	systemMessageMode := "system"
	if isReasoningModel {
		systemMessageMode = "developer"
	}

	return LanguageModelCapabilities{
		SupportsFlexProcessing:         supportsFlexProcessing,
		SupportsPriorityProcessing:     supportsPriorityProcessing,
		IsReasoningModel:               isReasoningModel,
		SystemMessageMode:              systemMessageMode,
		SupportsNonReasoningParameters: supportsNonReasoningParameters,
	}
}
