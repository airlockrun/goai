package cohere

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/model"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/cohere/src/*.test.ts

func createTestProvider(serverURL string) *Provider {
	return New(Options{
		APIKey:  "test-api-key",
		BaseURL: serverURL,
	})
}

func TestCohereProvider_ID(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	if provider.ID() != "cohere" {
		t.Errorf("expected provider ID cohere, got %s", provider.ID())
	}
}

func TestCohereProvider_Models(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})

	models := provider.Models()
	if len(models) == 0 {
		t.Error("expected at least one model")
	}

	hasCommandR := false
	for _, m := range models {
		if m == "command-r-plus" {
			hasCommandR = true
		}
	}
	if !hasCommandR {
		t.Error("expected command-r-plus in models list")
	}
}

// Chat model tests
func TestCohereModel_StreamText(t *testing.T) {
	t.Run("should extract text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/stream+json")
			w.WriteHeader(http.StatusOK)

			// Cohere uses newline-delimited JSON (NDJSON), not SSE
			chunks := []string{
				`{"event_type":"text-generation","text":"Hello"}`,
				`{"event_type":"text-generation","text":", "}`,
				`{"event_type":"text-generation","text":"World!"}`,
				`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"text":"Hello, World!","meta":{"tokens":{"input_tokens":5,"output_tokens":3}}}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("command-r-plus")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var textParts []string
		for event := range events {
			if event.Type == stream.EventTextDelta {
				if delta, ok := event.Data.(stream.TextDeltaEvent); ok {
					textParts = append(textParts, delta.Text)
				}
			}
		}

		text := strings.Join(textParts, "")
		if text != "Hello, World!" {
			t.Errorf("expected text 'Hello, World!', got %s", text)
		}
	})

	t.Run("should extract usage", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/stream+json")
			w.WriteHeader(http.StatusOK)

			// Cohere uses newline-delimited JSON (NDJSON), not SSE
			chunks := []string{
				`{"event_type":"text-generation","text":"Hi"}`,
				`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"text":"Hi","meta":{"tokens":{"input_tokens":10,"output_tokens":5}}}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("command-r-plus")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var usage stream.Usage
		for event := range events {
			if event.Type == stream.EventFinish {
				if finish, ok := event.Data.(stream.FinishEvent); ok {
					usage = finish.Usage
				}
			}
		}

		if usage.InputTotal() != 10 {
			t.Errorf("expected prompt tokens 10, got %d", usage.InputTotal())
		}
		if usage.OutputTotal() != 5 {
			t.Errorf("expected completion tokens 5, got %d", usage.OutputTotal())
		}
	})
}

func TestCohereModel_Headers(t *testing.T) {
	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "application/stream+json")
			w.WriteHeader(http.StatusOK)

			// Cohere uses newline-delimited JSON (NDJSON), not SSE
			chunks := []string{
				`{"event_type":"text-generation","text":"Hi"}`,
				`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"text":"Hi","meta":{"tokens":{"input_tokens":5,"output_tokens":1}}}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n"))
			}
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		model := provider.Model("command-r-plus")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Provider-Header") != "provider-header-value" {
			t.Errorf("expected Custom-Provider-Header, got %s", receivedHeaders.Get("Custom-Provider-Header"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

// Embedding model tests
var dummyEmbeddings = [][]float64{
	{0.1, 0.2, 0.3, 0.4, 0.5},
	{0.6, 0.7, 0.8, 0.9, 1.0},
}
var testEmbedValues = []string{"sunny day at the beach", "rainy day in the city"}

func TestCohereEmbedding_DoEmbed(t *testing.T) {
	t.Run("should extract embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id":         "emb-123",
				"embeddings": dummyEmbeddings,
				"texts":      testEmbedValues,
				"meta": map[string]any{
					"billed_units": map[string]any{
						"input_tokens": 10,
					},
				},
			})
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		embModel := provider.EmbeddingModel("embed-english-v3.0")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Embeddings) != 2 {
			t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
		}

		for i, emb := range result.Embeddings {
			for j, v := range emb.Values {
				if v != dummyEmbeddings[i][j] {
					t.Errorf("expected embedding[%d][%d] = %f, got %f", i, j, dummyEmbeddings[i][j], v)
				}
			}
		}
	})

	t.Run("should extract usage", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id":         "emb-123",
				"embeddings": [][]float64{dummyEmbeddings[0]},
				"texts":      testEmbedValues[:1],
				"meta": map[string]any{
					"billed_units": map[string]any{
						"input_tokens": 20,
					},
				},
			})
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		embModel := provider.EmbeddingModel("embed-english-v3.0")

		result, err := embModel.Embed(context.Background(), model.EmbedCallOptions{
			Values: testEmbedValues[:1],
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.Usage.Tokens != 20 {
			t.Errorf("expected usage tokens 20, got %d", result.Usage.Tokens)
		}
	})
}

// Reranking model tests
func TestCohereReranking_DoRerank(t *testing.T) {
	t.Run("should rerank documents", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id": "rerank-123",
				"results": []map[string]any{
					{"index": 1, "relevance_score": 0.95, "document": map[string]any{"text": "doc2"}},
					{"index": 0, "relevance_score": 0.75, "document": map[string]any{"text": "doc1"}},
				},
				"meta": map[string]any{
					"billed_units": map[string]any{
						"search_units": 1,
					},
				},
			})
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		rerankModel := provider.RerankingModel("rerank-english-v3.0")

		result, err := rerankModel.Rerank(context.Background(), model.RerankCallOptions{
			Query:     "What is the weather?",
			Documents: []string{"doc1", "doc2"},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Results) != 2 {
			t.Errorf("expected 2 results, got %d", len(result.Results))
		}

		// First result should be doc2 (higher score)
		if result.Results[0].Index != 1 {
			t.Errorf("expected first result index 1, got %d", result.Results[0].Index)
		}
		if result.Results[0].Score != 0.95 {
			t.Errorf("expected first result score 0.95, got %f", result.Results[0].Score)
		}

		if result.Usage.SearchUnits != 1 {
			t.Errorf("expected search units 1, got %d", result.Usage.SearchUnits)
		}
	})

	t.Run("should pass topN and returnDocuments", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id": "rerank-123",
				"results": []map[string]any{
					{"index": 0, "relevance_score": 0.95, "document": map[string]any{"text": "doc1"}},
				},
			})
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		rerankModel := provider.RerankingModel("rerank-english-v3.0")

		_, err := rerankModel.Rerank(context.Background(), model.RerankCallOptions{
			Query:           "What is the weather?",
			Documents:       []string{"doc1", "doc2", "doc3"},
			TopN:            1,
			ReturnDocuments: true,
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedBody["top_n"] != float64(1) {
			t.Errorf("expected top_n 1, got %v", receivedBody["top_n"])
		}
		if receivedBody["return_documents"] != true {
			t.Errorf("expected return_documents true, got %v", receivedBody["return_documents"])
		}
	})

	t.Run("should pass headers", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id":      "rerank-123",
				"results": []map[string]any{},
			})
		}))
		defer server.Close()

		provider := New(Options{
			APIKey:  "test-api-key",
			BaseURL: server.URL,
			Headers: map[string]string{
				"Custom-Provider-Header": "provider-header-value",
			},
		})
		rerankModel := provider.RerankingModel("rerank-english-v3.0")

		_, err := rerankModel.Rerank(context.Background(), model.RerankCallOptions{
			Query:     "test query",
			Documents: []string{"doc1"},
			Headers: map[string]string{
				"Custom-Request-Header": "request-header-value",
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if receivedHeaders.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("expected Authorization header, got %s", receivedHeaders.Get("Authorization"))
		}

		if receivedHeaders.Get("Custom-Request-Header") != "request-header-value" {
			t.Errorf("expected Custom-Request-Header, got %s", receivedHeaders.Get("Custom-Request-Header"))
		}
	})
}

func TestCohereReranking_MaxDocumentsPerCall(t *testing.T) {
	provider := New(Options{APIKey: "test-key"})
	rerankModel := provider.RerankingModel("rerank-english-v3.0")

	if rerankModel.MaxDocumentsPerCall() != 1000 {
		t.Errorf("expected max documents 1000, got %d", rerankModel.MaxDocumentsPerCall())
	}
}

// Tests for ProviderOptions - verifies ChatOptions are wired up correctly

func TestCohereModel_ThinkingConfig(t *testing.T) {
	t.Run("should pass thinking config", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "application/stream+json")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"event_type":"text-generation","text":"Hi"}`,
				`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"text":"Hi","meta":{"tokens":{"input_tokens":5,"output_tokens":1}}}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("command-r-plus")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"thinking": map[string]any{
					"type":        "enabled",
					"tokenBudget": 2048,
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		thinking, ok := receivedBody["thinking"].(map[string]any)
		if !ok {
			t.Fatal("expected thinking in request body")
		}

		if thinking["type"] != "enabled" {
			t.Errorf("expected type 'enabled', got %v", thinking["type"])
		}
		if thinking["token_budget"] != float64(2048) {
			t.Errorf("expected token_budget 2048, got %v", thinking["token_budget"])
		}
	})

	t.Run("should pass disabled thinking", func(t *testing.T) {
		var receivedBody map[string]any

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &receivedBody)

			w.Header().Set("Content-Type", "application/stream+json")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				`{"event_type":"text-generation","text":"Hi"}`,
				`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"text":"Hi","meta":{"tokens":{"input_tokens":5,"output_tokens":1}}}}`,
			}

			for _, chunk := range chunks {
				w.Write([]byte(chunk + "\n"))
			}
		}))
		defer server.Close()

		provider := createTestProvider(server.URL)
		model := provider.Model("command-r-plus")

		events, err := model.Stream(context.Background(), &stream.CallOptions{
			Messages: []message.Message{
				message.NewUserMessage("Hello"),
			},
			ProviderOptions: map[string]any{
				"thinking": map[string]any{
					"type": "disabled",
				},
			},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Drain events
		for range events {
		}

		thinking, ok := receivedBody["thinking"].(map[string]any)
		if !ok {
			t.Fatal("expected thinking in request body")
		}

		if thinking["type"] != "disabled" {
			t.Errorf("expected type 'disabled', got %v", thinking["type"])
		}
	})
}

func TestCohereModel_ResponseFormat(t *testing.T) {
	runCapture := func(t *testing.T, callOpts *stream.CallOptions) map[string]any {
		t.Helper()
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"meta":{"tokens":{"input_tokens":1,"output_tokens":1}}}}` + "\n"))
		}))
		defer server.Close()
		prov := createTestProvider(server.URL)
		m := prov.Model("command-r")
		if callOpts.Messages == nil {
			callOpts.Messages = []message.Message{message.NewUserMessage("hi")}
		}
		events, err := m.Stream(context.Background(), callOpts)
		if err != nil {
			t.Fatal(err)
		}
		for range events {
		}
		return body
	}

	t.Run("json without schema", func(t *testing.T) {
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json"},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Errorf("expected json_object type, got %v", rf)
		}
		if _, has := rf["json_schema"]; has {
			t.Errorf("expected no json_schema when no schema given, got %v", rf)
		}
	})

	t.Run("json with schema keeps type json_object with embedded schema", func(t *testing.T) {
		schema := json.RawMessage(`{"type":"object"}`)
		body := runCapture(t, &stream.CallOptions{
			ResponseFormat: &stream.ResponseFormat{Type: "json", Schema: schema},
		})
		rf, _ := body["response_format"].(map[string]any)
		if rf["type"] != "json_object" {
			t.Errorf("expected json_object type, got %v", rf)
		}
		if _, has := rf["json_schema"]; !has {
			t.Errorf("expected json_schema to be present, got %v", rf)
		}
	})
}

// Regression for ai-sdk #0df64d6: outputDimension providerOption on the
// Cohere Embed API maps to `output_dimension` on the wire. Only embed-v4
// models honor it (Cohere returns an error for older families), but the
// transport should forward it verbatim.
func TestCohereEmbedding_OutputDimensionOnWire(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"e","embeddings":[[0.1]]}`))
	}))
	defer server.Close()

	p := createTestProvider(server.URL)
	em := p.EmbeddingModel("embed-v4.0")
	if em == nil {
		t.Fatal("embedding model not available")
	}
	_, err := em.Embed(context.Background(), model.EmbedCallOptions{
		Values: []string{"hello"},
		ProviderOptions: map[string]any{
			"outputDimension": 512,
			"inputType":       "search_query",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if captured["output_dimension"] != float64(512) {
		t.Errorf("output_dimension = %v, want 512", captured["output_dimension"])
	}
	if captured["input_type"] != "search_query" {
		t.Errorf("input_type = %v, want search_query", captured["input_type"])
	}
}
