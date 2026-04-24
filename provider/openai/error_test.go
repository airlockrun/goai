package openai

import (
	"testing"
)

// Tests for OpenAI error parsing - translated from ai-sdk
// Source: ai-sdk/packages/openai/src/openai-error.test.ts

func TestParseOpenAIError(t *testing.T) {
	t.Run("should parse OpenRouter resource exhausted error", func(t *testing.T) {
		// This is a real error from OpenRouter that wraps the underlying
		// provider's error in the message field as escaped JSON.
		input := `{"error":{"message":"{\n  \"error\": {\n    \"code\": 429,\n    \"message\": \"Resource has been exhausted (e.g. check quota).\",\n    \"status\": \"RESOURCE_EXHAUSTED\"\n  }\n}\n","code":429}}`

		result, err := ParseOpenAIError([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// The message should contain the nested JSON as a string
		expectedMessage := "{\n  \"error\": {\n    \"code\": 429,\n    \"message\": \"Resource has been exhausted (e.g. check quota).\",\n    \"status\": \"RESOURCE_EXHAUSTED\"\n  }\n}\n"
		if result.Error.Message != expectedMessage {
			t.Errorf("unexpected message:\ngot:  %q\nwant: %q", result.Error.Message, expectedMessage)
		}

		// The code should be 429
		if result.GetCodeNumber() != 429 {
			t.Errorf("unexpected code: got %d, want 429", result.GetCodeNumber())
		}
	})

	t.Run("should parse standard OpenAI error", func(t *testing.T) {
		input := `{"error":{"message":"Invalid API key","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}`

		result, err := ParseOpenAIError([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Error.Message != "Invalid API key" {
			t.Errorf("unexpected message: %s", result.Error.Message)
		}
		if result.GetType() != "invalid_request_error" {
			t.Errorf("unexpected type: %s", result.GetType())
		}
		if result.GetCodeString() != "invalid_api_key" {
			t.Errorf("unexpected code: %s", result.GetCodeString())
		}
	})

	t.Run("should handle missing optional fields", func(t *testing.T) {
		input := `{"error":{"message":"Something went wrong"}}`

		result, err := ParseOpenAIError([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Error.Message != "Something went wrong" {
			t.Errorf("unexpected message: %s", result.Error.Message)
		}
		if result.GetType() != "" {
			t.Errorf("expected empty type, got: %s", result.GetType())
		}
		if result.GetCodeString() != "" {
			t.Errorf("expected empty code string, got: %s", result.GetCodeString())
		}
		if result.GetCodeNumber() != 0 {
			t.Errorf("expected code number 0, got: %d", result.GetCodeNumber())
		}
	})

	t.Run("should handle numeric code", func(t *testing.T) {
		input := `{"error":{"message":"Rate limited","code":429}}`

		result, err := ParseOpenAIError([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.GetCodeNumber() != 429 {
			t.Errorf("expected code 429, got: %d", result.GetCodeNumber())
		}
		// Code as string should be empty for numeric codes
		if result.GetCodeString() != "" {
			t.Errorf("expected empty code string for numeric code, got: %s", result.GetCodeString())
		}
	})

	t.Run("should handle string code", func(t *testing.T) {
		input := `{"error":{"message":"Context length exceeded","code":"context_length_exceeded"}}`

		result, err := ParseOpenAIError([]byte(input))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.GetCodeString() != "context_length_exceeded" {
			t.Errorf("expected code 'context_length_exceeded', got: %s", result.GetCodeString())
		}
		// Code as number should be 0 for string codes
		if result.GetCodeNumber() != 0 {
			t.Errorf("expected code number 0 for string code, got: %d", result.GetCodeNumber())
		}
	})

	t.Run("should return error for invalid JSON", func(t *testing.T) {
		input := `not valid json`

		_, err := ParseOpenAIError([]byte(input))
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}

func TestOpenAIErrorData_GetMessage(t *testing.T) {
	data := &OpenAIErrorData{
		Error: OpenAIErrorInfo{
			Message: "Test error message",
		},
	}

	if data.GetMessage() != "Test error message" {
		t.Errorf("unexpected message: %s", data.GetMessage())
	}
}
