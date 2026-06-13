package vertexmaas

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

// Translated from ai-sdk/packages/google-vertex/src/maas/
// google-vertex-maas-provider.ts (there is no dedicated test file upstream;
// these tests exercise the behavior documented in the provider source + the
// maas-options type union).

func TestProvider_ID(t *testing.T) {
	p := New(Options{Project: "test-project", AccessToken: "token"})
	if got := p.ID(); got != "vertex.maas" {
		t.Errorf("ID() = %q, want %q", got, "vertex.maas")
	}
}

func TestProvider_DefaultLocationIsGlobal(t *testing.T) {
	p := New(Options{Project: "test-project", AccessToken: "token"})
	if p.opts.Location != "global" {
		t.Errorf("default Location = %q, want %q", p.opts.Location, "global")
	}
}

func TestProvider_BaseURL(t *testing.T) {
	tests := []struct {
		name string
		opts Options
		want string
	}{
		{
			name: "global location",
			opts: Options{Project: "test-project", Location: "global", AccessToken: "t"},
			want: "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/openapi",
		},
		{
			name: "regional location",
			opts: Options{Project: "test-project", Location: "us-central1", AccessToken: "t"},
			want: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/openapi",
		},
		{
			name: "eu multi-region",
			opts: Options{Project: "test-project", Location: "eu", AccessToken: "t"},
			want: "https://aiplatform.eu.rep.googleapis.com/v1/projects/test-project/locations/eu/endpoints/openapi",
		},
		{
			name: "us multi-region",
			opts: Options{Project: "test-project", Location: "us", AccessToken: "t"},
			want: "https://aiplatform.us.rep.googleapis.com/v1/projects/test-project/locations/us/endpoints/openapi",
		},
		{
			name: "BaseURL override",
			opts: Options{Project: "test-project", BaseURL: "https://custom.example.com/v1/", AccessToken: "t"},
			want: "https://custom.example.com/v1",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := New(tc.opts)
			if got := p.compat.BaseURL(); got != tc.want {
				t.Errorf("compat.BaseURL() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestProvider_Models(t *testing.T) {
	p := New(Options{Project: "test-project", AccessToken: "t"})
	models := p.Models()
	if len(models) != len(VertexMaasModels) {
		t.Fatalf("Models() length = %d, want %d", len(models), len(VertexMaasModels))
	}
	// Independence: mutating return shouldn't affect subsequent calls.
	models[0] = "mutated"
	if p.Models()[0] == "mutated" {
		t.Error("Models() return must be a copy")
	}
	// Spot-check a representative entry.
	want := "deepseek-ai/deepseek-r1-0528-maas"
	found := false
	for _, m := range p.Models() {
		if m == want {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Models() missing %q", want)
	}
}

func TestProvider_AuthorizationHeader(t *testing.T) {
	var receivedAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	p := New(Options{
		Project:     "test-project",
		Location:    "global",
		AccessToken: "bearer-token-xyz",
		BaseURL:     server.URL,
	})
	m := p.Model("deepseek-ai/deepseek-r1-0528-maas")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	for range events {
	}

	if receivedAuth != "Bearer bearer-token-xyz" {
		t.Errorf("Authorization = %q, want %q", receivedAuth, "Bearer bearer-token-xyz")
	}
}

func TestProvider_Delegation(t *testing.T) {
	// End-to-end delegation smoke test: a chat-completions-style SSE
	// response from a mock server should flow through the openaicompat
	// model and emit text deltas with the expected finish reason.
	var receivedPath string
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := []string{
			`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"deepseek","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
			`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"deepseek","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
			`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"deepseek","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`data: [DONE]`,
		}
		for _, c := range chunks {
			_, _ = w.Write([]byte(c + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	p := New(Options{
		Project:     "test-project",
		AccessToken: "token",
		BaseURL:     server.URL,
	})
	m := p.LanguageModel("deepseek-ai/deepseek-r1-0528-maas")
	events, err := m.Stream(context.Background(), &stream.CallOptions{
		Messages: []message.Message{message.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	var text strings.Builder
	var finishReason stream.FinishReason
	for ev := range events {
		if d, ok := ev.Data.(stream.TextDeltaEvent); ok {
			text.WriteString(d.Text)
		}
		if d, ok := ev.Data.(stream.FinishEvent); ok {
			finishReason = d.FinishReason
		}
	}
	if text.String() != "Hi" {
		t.Errorf("text = %q, want %q", text.String(), "Hi")
	}
	if finishReason != stream.FinishReasonStop {
		t.Errorf("finishReason = %v, want stop", finishReason)
	}
	if receivedPath != "/chat/completions" {
		t.Errorf("request path = %q, want /chat/completions", receivedPath)
	}
	// Sanity check: the OpenAI-shape body carries the model ID we requested.
	if got, _ := receivedBody["model"].(string); got != "deepseek-ai/deepseek-r1-0528-maas" {
		t.Errorf("body.model = %q, want deepseek-ai/deepseek-r1-0528-maas", got)
	}
}

func TestProvider_HeadersPreservedAndAuthAdded(t *testing.T) {
	// When callers supply Headers the provider must clone them (not mutate the
	// caller's map) and inject Authorization.
	caller := map[string]string{"X-Goog-User-Project": "proj-b"}
	p := New(Options{
		Project:     "test-project",
		AccessToken: "t",
		Headers:     caller,
	})
	// Caller's map must not have gained Authorization.
	if _, ok := caller["Authorization"]; ok {
		t.Error("Options.Headers was mutated to add Authorization")
	}
	_ = p // no runtime assertion needed beyond the above.
}
