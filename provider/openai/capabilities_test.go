package openai

import "testing"

// Tests for model capabilities - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/openai-language-model-capabilities.test.ts

func TestGetLanguageModelCapabilities_IsReasoningModel(t *testing.T) {
	testCases := []struct {
		modelID  string
		expected bool
	}{
		// Non-reasoning models (GPT-4.x series)
		{"gpt-4.1", false},
		{"gpt-4.1-2025-04-14", false},
		{"gpt-4.1-mini", false},
		{"gpt-4.1-mini-2025-04-14", false},
		{"gpt-4.1-nano", false},
		{"gpt-4.1-nano-2025-04-14", false},

		// Non-reasoning models (GPT-4o series)
		{"gpt-4o", false},
		{"gpt-4o-2024-05-13", false},
		{"gpt-4o-2024-08-06", false},
		{"gpt-4o-2024-11-20", false},
		{"gpt-4o-audio-preview", false},
		{"gpt-4o-audio-preview-2024-10-01", false},
		{"gpt-4o-audio-preview-2024-12-17", false},
		{"gpt-4o-search-preview", false},
		{"gpt-4o-search-preview-2025-03-11", false},
		{"gpt-4o-mini-search-preview", false},
		{"gpt-4o-mini-search-preview-2025-03-11", false},
		{"gpt-4o-mini", false},
		{"gpt-4o-mini-2024-07-18", false},

		// Non-reasoning models (GPT-4 turbo series)
		{"gpt-4-turbo", false},
		{"gpt-4-turbo-2024-04-09", false},
		{"gpt-4-turbo-preview", false},
		{"gpt-4-0125-preview", false},
		{"gpt-4-1106-preview", false},
		{"gpt-4", false},
		{"gpt-4-0613", false},

		// Non-reasoning models (GPT-4.5, GPT-3.5)
		{"gpt-4.5-preview", false},
		{"gpt-4.5-preview-2025-02-27", false},
		{"gpt-3.5-turbo-0125", false},
		{"gpt-3.5-turbo", false},
		{"gpt-3.5-turbo-1106", false},

		// Non-reasoning models (other)
		{"chatgpt-4o-latest", false},
		{"gpt-5-chat-latest", false}, // chat variant is NOT a reasoning model

		// Reasoning models (o1 series)
		{"o1", true},
		{"o1-2024-12-17", true},

		// Reasoning models (o3 series)
		{"o3-mini", true},
		{"o3-mini-2025-01-31", true},
		{"o3", true},
		{"o3-2025-04-16", true},

		// Reasoning models (o4 series)
		{"o4-mini", true},
		{"o4-mini-2025-04-16", true},

		// Reasoning models (codex, computer-use)
		{"codex-mini-latest", true},
		{"computer-use-preview", true},

		// Reasoning models (GPT-5 series - except chat variant)
		{"gpt-5", true},
		{"gpt-5-2025-08-07", true},
		{"gpt-5-codex", true},
		{"gpt-5-mini", true},
		{"gpt-5-mini-2025-08-07", true},
		{"gpt-5-nano", true},
		{"gpt-5-nano-2025-08-07", true},
		{"gpt-5-pro", true},
		{"gpt-5-pro-2025-10-06", true},

		// Unknown/custom models default to non-reasoning
		{"new-unknown-model", false},
		{"ft:gpt-4o-2024-08-06:org:custom:abc123", false},
		{"custom-model", false},
	}

	for _, tc := range testCases {
		t.Run(tc.modelID, func(t *testing.T) {
			caps := GetLanguageModelCapabilities(tc.modelID)
			if caps.IsReasoningModel != tc.expected {
				t.Errorf("IsReasoningModel for %s: got %v, want %v",
					tc.modelID, caps.IsReasoningModel, tc.expected)
			}
		})
	}
}

func TestGetLanguageModelCapabilities_SupportsNonReasoningParameters(t *testing.T) {
	testCases := []struct {
		modelID  string
		expected bool
	}{
		// Models that support non-reasoning parameters (gpt-5.1, gpt-5.2)
		{"gpt-5.1", true},
		{"gpt-5.1-chat-latest", true},
		{"gpt-5.1-codex-mini", true},
		{"gpt-5.1-codex", true},
		{"gpt-5.2", true},
		{"gpt-5.2-pro", true},
		{"gpt-5.2-chat-latest", true},

		// Models that don't support non-reasoning parameters
		{"gpt-5", false},
		{"gpt-5-mini", false},
		{"gpt-5-nano", false},
		{"gpt-5-pro", false},
		{"gpt-5-chat-latest", false},
	}

	for _, tc := range testCases {
		t.Run(tc.modelID, func(t *testing.T) {
			caps := GetLanguageModelCapabilities(tc.modelID)
			if caps.SupportsNonReasoningParameters != tc.expected {
				t.Errorf("SupportsNonReasoningParameters for %s: got %v, want %v",
					tc.modelID, caps.SupportsNonReasoningParameters, tc.expected)
			}
		})
	}
}

func TestGetLanguageModelCapabilities_SystemMessageMode(t *testing.T) {
	testCases := []struct {
		modelID  string
		expected string
	}{
		// Reasoning models use developer role
		{"o1", "developer"},
		{"o3-mini", "developer"},
		{"gpt-5", "developer"},

		// Non-reasoning models use system role
		{"gpt-4o", "system"},
		{"gpt-4-turbo", "system"},
		{"gpt-5-chat-latest", "system"},
	}

	for _, tc := range testCases {
		t.Run(tc.modelID, func(t *testing.T) {
			caps := GetLanguageModelCapabilities(tc.modelID)
			if caps.SystemMessageMode != tc.expected {
				t.Errorf("SystemMessageMode for %s: got %v, want %v",
					tc.modelID, caps.SystemMessageMode, tc.expected)
			}
		})
	}
}

func TestGetLanguageModelCapabilities_SupportsFlexProcessing(t *testing.T) {
	testCases := []struct {
		modelID  string
		expected bool
	}{
		// Models that support flex processing
		{"o3", true},
		{"o3-mini", true},
		{"o4-mini", true},
		{"gpt-5", true},
		{"gpt-5-mini", true},

		// Models that don't support flex processing
		{"gpt-4o", false},
		{"gpt-4-turbo", false},
		{"gpt-5-chat-latest", false}, // chat variant excluded
		{"o1", false},
	}

	for _, tc := range testCases {
		t.Run(tc.modelID, func(t *testing.T) {
			caps := GetLanguageModelCapabilities(tc.modelID)
			if caps.SupportsFlexProcessing != tc.expected {
				t.Errorf("SupportsFlexProcessing for %s: got %v, want %v",
					tc.modelID, caps.SupportsFlexProcessing, tc.expected)
			}
		})
	}
}

func TestGetLanguageModelCapabilities_SupportsPriorityProcessing(t *testing.T) {
	testCases := []struct {
		modelID  string
		expected bool
	}{
		// Models that support priority processing
		{"gpt-4o", true},
		{"gpt-4-turbo", true},
		{"o3", true},
		{"o3-mini", true},
		{"o4-mini", true},
		{"gpt-5", true},
		{"gpt-5-mini", true},
		{"gpt-5-pro", true},

		// Models that don't support priority processing
		{"gpt-3.5-turbo", false},
		{"gpt-5-nano", false},        // nano excluded
		{"gpt-5-chat-latest", false}, // chat variant excluded
		{"o1", false},
	}

	for _, tc := range testCases {
		t.Run(tc.modelID, func(t *testing.T) {
			caps := GetLanguageModelCapabilities(tc.modelID)
			if caps.SupportsPriorityProcessing != tc.expected {
				t.Errorf("SupportsPriorityProcessing for %s: got %v, want %v",
					tc.modelID, caps.SupportsPriorityProcessing, tc.expected)
			}
		})
	}
}
