package openai

import (
	"regexp"
	"strconv"
	"strings"
)

var oSeriesPattern = regexp.MustCompile(`^o(\d+)(?:-|$)`)
var gptPattern = regexp.MustCompile(`^gpt-(\d+)(?:\.(\d+))?(?:-(.+))?$`)

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
	o := oSeriesPattern.FindStringSubmatch(modelID)
	gpt := gptPattern.FindStringSubmatch(modelID)
	var major, minor, oVersion int
	var chat, nano bool
	if o != nil {
		oVersion, _ = strconv.Atoi(o[1])
	}
	if gpt != nil {
		major, _ = strconv.Atoi(gpt[1])
		minor, _ = strconv.Atoi(gpt[2])
		chat = gpt[2] == "" && strings.HasPrefix(gpt[3], "chat")
		nano = strings.HasPrefix(gpt[3], "nano")
	}
	supportsFlexProcessing := oVersion >= 3 || major >= 5 && !chat
	supportsPriorityProcessing := strings.HasPrefix(modelID, "gpt-4") || oVersion >= 3 || major >= 5 && !nano && !chat
	isReasoningModel := o != nil || major >= 5 && !chat
	supportsNonReasoningParameters := major > 5 || major == 5 && minor >= 1

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
