package response

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/schema"
)

// Test types for JSON parsing
type testPerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

// mockResponse creates a mock http.Response with the given body and status.
func mockResponse(body string, statusCode int, statusText string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Status:     statusText,
		Body:       io.NopCloser(bytes.NewBufferString(body)),
		Header:     make(http.Header),
	}
}

// TestCreateJSONResponseHandler_ParsesAndValidates tests that the handler
// correctly parses JSON and returns both value and rawValue.
// Source: ai-sdk/packages/provider-utils/src/response-handler.test.ts
func TestCreateJSONResponseHandler_ParsesAndValidates(t *testing.T) {
	rawData := map[string]any{
		"name":       "John",
		"age":        float64(30), // JSON numbers are float64
		"extraField": "ignored",
	}
	body, _ := json.Marshal(rawData)

	handler := CreateJSONResponseHandler[testPerson](nil)

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse(string(body), 200, "OK"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Value.Name != "John" {
		t.Errorf("expected name 'John', got %q", result.Value.Name)
	}
	if result.Value.Age != 30 {
		t.Errorf("expected age 30, got %d", result.Value.Age)
	}

	// rawValue should contain all fields including extraField
	rawMap, ok := result.RawValue.(map[string]any)
	if !ok {
		t.Fatalf("expected rawValue to be map, got %T", result.RawValue)
	}
	if rawMap["extraField"] != "ignored" {
		t.Errorf("expected rawValue to contain extraField, got %v", rawMap)
	}
}

// TestCreateJSONResponseHandler_WithSchema tests JSON parsing with a schema.
func TestCreateJSONResponseHandler_WithSchema(t *testing.T) {
	personSchema := schema.Object(map[string]*schema.Schema{
		"name": schema.String(),
		"age":  schema.Integer(),
	})

	body := `{"name": "Alice", "age": 25}`

	handler := CreateJSONResponseHandler[testPerson](personSchema)

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse(body, 200, "OK"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Value.Name != "Alice" {
		t.Errorf("expected name 'Alice', got %q", result.Value.Name)
	}
	if result.Value.Age != 25 {
		t.Errorf("expected age 25, got %d", result.Value.Age)
	}
}

// TestCreateJSONResponseHandler_InvalidJSON tests error handling for invalid JSON.
func TestCreateJSONResponseHandler_InvalidJSON(t *testing.T) {
	handler := CreateJSONResponseHandler[testPerson](nil)

	_, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse("not json", 200, "OK"),
	})

	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}

	var apiErr *errors.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APICallError, got %T", err)
	}

	if apiErr.Message != "Invalid JSON response" {
		t.Errorf("expected message 'Invalid JSON response', got %q", apiErr.Message)
	}
}

// TestCreateBinaryResponseHandler_HandlesBinaryResponse tests binary response handling.
// Source: ai-sdk/packages/provider-utils/src/response-handler.test.ts
func TestCreateBinaryResponseHandler_HandlesBinaryResponse(t *testing.T) {
	binaryData := []byte{1, 2, 3, 4}
	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(binaryData)),
		Header:     make(http.Header),
	}

	handler := CreateBinaryResponseHandler()

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          resp,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !bytes.Equal(result.Value, binaryData) {
		t.Errorf("expected %v, got %v", binaryData, result.Value)
	}
}

// TestCreateBinaryResponseHandler_EmptyBody tests error for empty body.
// Source: ai-sdk/packages/provider-utils/src/response-handler.test.ts
func TestCreateBinaryResponseHandler_EmptyBody(t *testing.T) {
	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader([]byte{})),
		Header:     make(http.Header),
	}

	handler := CreateBinaryResponseHandler()

	_, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          resp,
	})

	if err == nil {
		t.Fatal("expected error for empty body")
	}

	var apiErr *errors.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APICallError, got %T", err)
	}

	if apiErr.Message != "Response body is empty" {
		t.Errorf("expected message 'Response body is empty', got %q", apiErr.Message)
	}
}

// TestCreateBinaryResponseHandler_NilBody tests error for nil body.
func TestCreateBinaryResponseHandler_NilBody(t *testing.T) {
	resp := &http.Response{
		StatusCode: 200,
		Body:       nil,
		Header:     make(http.Header),
	}

	handler := CreateBinaryResponseHandler()

	_, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          resp,
	})

	if err == nil {
		t.Fatal("expected error for nil body")
	}

	var apiErr *errors.APICallError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APICallError, got %T", err)
	}
}

// TestCreateStatusCodeErrorResponseHandler tests status code error handling.
// Source: ai-sdk/packages/provider-utils/src/response-handler.test.ts
func TestCreateStatusCodeErrorResponseHandler(t *testing.T) {
	handler := CreateStatusCodeErrorResponseHandler()

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{"some": "data"},
		Response:          mockResponse("Error message", 404, "404 Not Found"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	apiErr := result.Value

	if apiErr.Message != "404 Not Found" {
		t.Errorf("expected message '404 Not Found', got %q", apiErr.Message)
	}
	if apiErr.StatusCode != 404 {
		t.Errorf("expected status 404, got %d", apiErr.StatusCode)
	}
	if apiErr.ResponseBody != "Error message" {
		t.Errorf("expected body 'Error message', got %q", apiErr.ResponseBody)
	}
	if apiErr.URL != "test-url" {
		t.Errorf("expected url 'test-url', got %q", apiErr.URL)
	}

	reqBody, ok := apiErr.RequestBodyValues.(map[string]any)
	if !ok || reqBody["some"] != "data" {
		t.Errorf("expected requestBodyValues {some: data}, got %v", apiErr.RequestBodyValues)
	}
}

// Test error data type for JSON error handler tests
type testErrorData struct {
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// TestCreateJSONErrorResponseHandler_ParsesProviderError tests JSON error parsing.
func TestCreateJSONErrorResponseHandler_ParsesProviderError(t *testing.T) {
	handler := CreateJSONErrorResponseHandler(JSONErrorConfig[testErrorData]{
		ErrorSchema: func(body []byte) (testErrorData, error) {
			var data testErrorData
			err := json.Unmarshal(body, &data)
			return data, err
		},
		ErrorToMessage: func(data testErrorData) string {
			return data.Error.Message
		},
		IsRetryable: func(resp *http.Response, data *testErrorData) bool {
			return resp.StatusCode == 429 || resp.StatusCode >= 500
		},
	})

	errorBody := `{"error": {"message": "Rate limit exceeded", "code": "rate_limit"}}`

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse(errorBody, 429, "429 Too Many Requests"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	apiErr := result.Value

	if apiErr.Message != "Rate limit exceeded" {
		t.Errorf("expected message 'Rate limit exceeded', got %q", apiErr.Message)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", apiErr.StatusCode)
	}
	if !apiErr.IsRetryable {
		t.Error("expected error to be retryable")
	}

	// Data should contain parsed error
	if apiErr.Data == nil {
		t.Fatal("expected Data to be populated")
	}
	parsedData, ok := apiErr.Data.(testErrorData)
	if !ok {
		t.Fatalf("expected Data to be testErrorData, got %T", apiErr.Data)
	}
	if parsedData.Error.Code != "rate_limit" {
		t.Errorf("expected code 'rate_limit', got %q", parsedData.Error.Code)
	}
}

// TestCreateJSONErrorResponseHandler_EmptyBody tests fallback to status text.
func TestCreateJSONErrorResponseHandler_EmptyBody(t *testing.T) {
	handler := CreateJSONErrorResponseHandler(JSONErrorConfig[testErrorData]{
		ErrorSchema: func(body []byte) (testErrorData, error) {
			var data testErrorData
			err := json.Unmarshal(body, &data)
			return data, err
		},
		ErrorToMessage: func(data testErrorData) string {
			return data.Error.Message
		},
	})

	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse("", 500, "500 Internal Server Error"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	apiErr := result.Value

	// Should fall back to status text
	if apiErr.Message != "500 Internal Server Error" {
		t.Errorf("expected message '500 Internal Server Error', got %q", apiErr.Message)
	}
}

// TestCreateJSONErrorResponseHandler_InvalidJSON tests fallback when JSON doesn't match schema.
func TestCreateJSONErrorResponseHandler_InvalidJSON(t *testing.T) {
	handler := CreateJSONErrorResponseHandler(JSONErrorConfig[testErrorData]{
		ErrorSchema: func(body []byte) (testErrorData, error) {
			var data testErrorData
			err := json.Unmarshal(body, &data)
			return data, err
		},
		ErrorToMessage: func(data testErrorData) string {
			return data.Error.Message
		},
	})

	// Invalid JSON that doesn't match schema
	result, err := handler(HandlerOptions{
		URL:               "test-url",
		RequestBodyValues: map[string]any{},
		Response:          mockResponse("not valid json", 400, "400 Bad Request"),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	apiErr := result.Value

	// Should fall back to status text
	if apiErr.Message != "400 Bad Request" {
		t.Errorf("expected message '400 Bad Request', got %q", apiErr.Message)
	}
}

// TestExtractResponseHeaders tests header extraction.
func TestExtractResponseHeaders(t *testing.T) {
	resp := &http.Response{
		Header: http.Header{
			"Content-Type":   []string{"application/json"},
			"X-Request-Id":   []string{"abc123"},
			"Multiple-Value": []string{"first", "second"},
		},
	}

	headers := ExtractResponseHeaders(resp)

	if headers["Content-Type"] != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", headers["Content-Type"])
	}
	if headers["X-Request-Id"] != "abc123" {
		t.Errorf("expected X-Request-Id 'abc123', got %q", headers["X-Request-Id"])
	}
	// Only first value should be returned for multi-value headers
	if headers["Multiple-Value"] != "first" {
		t.Errorf("expected Multiple-Value 'first', got %q", headers["Multiple-Value"])
	}
}

// TestExtractResponseHeaders_NilResponse tests nil response handling.
func TestExtractResponseHeaders_NilResponse(t *testing.T) {
	headers := ExtractResponseHeaders(nil)
	if headers != nil {
		t.Errorf("expected nil for nil response, got %v", headers)
	}
}

// TestSafeParseJSON tests safe JSON parsing.
func TestSafeParseJSON(t *testing.T) {
	t.Run("valid JSON", func(t *testing.T) {
		result := SafeParseJSON[testPerson](`{"name": "Bob", "age": 35}`, nil)
		if !result.Success {
			t.Fatalf("expected success, got error: %v", result.Error)
		}
		if result.Value.Name != "Bob" {
			t.Errorf("expected name 'Bob', got %q", result.Value.Name)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		result := SafeParseJSON[testPerson]("not json", nil)
		if result.Success {
			t.Fatal("expected failure for invalid JSON")
		}
		if result.Error == nil {
			t.Fatal("expected error to be set")
		}
	})

	t.Run("rawValue preserved", func(t *testing.T) {
		result := SafeParseJSON[testPerson](`{"name": "Test", "age": 1, "extra": true}`, nil)
		if !result.Success {
			t.Fatalf("expected success, got error: %v", result.Error)
		}
		rawMap, ok := result.RawValue.(map[string]any)
		if !ok {
			t.Fatalf("expected rawValue to be map, got %T", result.RawValue)
		}
		if rawMap["extra"] != true {
			t.Errorf("expected extra field in rawValue")
		}
	})
}

// TestParseJSON tests JSON parsing with errors.
func TestParseJSON(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		value, err := ParseJSON[testPerson](`{"name": "Test", "age": 1}`, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if value.Name != "Test" {
			t.Errorf("expected name 'Test', got %q", value.Name)
		}
	})

	t.Run("error", func(t *testing.T) {
		_, err := ParseJSON[testPerson]("invalid", nil)
		if err == nil {
			t.Fatal("expected error for invalid JSON")
		}
	})
}
