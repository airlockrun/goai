package provider_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/provider/azure"
	"github.com/airlockrun/goai/provider/cerebras"
	"github.com/airlockrun/goai/provider/cohere"
	"github.com/airlockrun/goai/provider/deepseek"
	"github.com/airlockrun/goai/provider/fireworks"
	"github.com/airlockrun/goai/provider/google"
	"github.com/airlockrun/goai/provider/groq"
	"github.com/airlockrun/goai/provider/mistral"
	"github.com/airlockrun/goai/provider/openai"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/provider/openresponses"
	"github.com/airlockrun/goai/provider/vertex"
	"github.com/airlockrun/goai/provider/xai"
	"github.com/airlockrun/goai/stream"
)

func TestReasoningRequests(t *testing.T) {
	models := map[string]func(string, string) stream.Model{
		"openresponses": func(url, id string) stream.Model {
			return openresponses.New(openresponses.Options{BaseURL: url}).Model(id)
		},
		"openai-chat":      func(url, id string) stream.Model { return openai.New(provider.Options{BaseURL: url}).Chat(id) },
		"openai-responses": func(url, id string) stream.Model { return openai.New(provider.Options{BaseURL: url}).Responses(id) },
		"azure-chat":       func(url, id string) stream.Model { return azure.New(azure.Options{BaseURL: url}).Chat(id) },
		"azure-responses":  func(url, id string) stream.Model { return azure.New(azure.Options{BaseURL: url}).Responses(id) },
		"compat": func(url, id string) stream.Model {
			return openaicompat.New(openaicompat.Options{BaseURL: url}).Model(id)
		},
		"anthropic":     func(url, id string) stream.Model { return anthropic.New(anthropic.Options{BaseURL: url}).Model(id) },
		"google":        func(url, id string) stream.Model { return google.New(google.Options{BaseURL: url}).Model(id) },
		"vertex":        func(url, id string) stream.Model { return vertex.New(vertex.Options{BaseURL: url}).Model(id) },
		"xai-chat":      func(url, id string) stream.Model { return xai.New(xai.Options{BaseURL: url}).Chat(id) },
		"xai-responses": func(url, id string) stream.Model { return xai.New(xai.Options{BaseURL: url}).Responses(id) },
		"deepseek":      func(url, id string) stream.Model { return deepseek.New(deepseek.Options{BaseURL: url}).Model(id) },
		"groq":          func(url, id string) stream.Model { return groq.New(groq.Options{BaseURL: url}).Model(id) },
		"mistral":       func(url, id string) stream.Model { return mistral.New(mistral.Options{BaseURL: url}).Model(id) },
		"cerebras":      func(url, id string) stream.Model { return cerebras.New(cerebras.Options{BaseURL: url}).Model(id) },
		"fireworks":     func(url, id string) stream.Model { return fireworks.New(fireworks.Options{BaseURL: url}).Model(id) },
		"cohere":        func(url, id string) stream.Model { return cohere.New(cohere.Options{BaseURL: url}).Model(id) },
	}
	for _, tc := range []struct {
		provider, model, effort string
		explicit                map[string]any
		want                    map[string]any
		warning                 string
	}{
		{"openai-chat", "gpt-5", "high", nil, map[string]any{"reasoning_effort": "high"}, ""},
		{"openai-chat", "gpt-4o", "high", nil, map[string]any{"reasoning_effort": nil}, "unsupported"},
		{"openai-chat", "gpt-5", "provider-default", nil, map[string]any{"reasoning_effort": nil}, ""},
		{"openai-responses", "gpt-5", "none", nil, map[string]any{"reasoning.effort": "none"}, ""},
		{"openresponses", "model", "minimal", nil, map[string]any{"reasoning.effort": "low"}, "compatibility"},
		{"openai-responses", "gpt-4o", "high", nil, map[string]any{"reasoning": nil}, "unsupported"},
		{"openai-responses", "gpt-5", "provider-default", nil, map[string]any{"reasoning": nil}, ""},
		{"azure-chat", "deployment", "high", nil, map[string]any{"reasoning_effort": "high"}, ""},
		{"azure-chat", "deployment", "low", map[string]any{"reasoningEffort": "high"}, map[string]any{"reasoning_effort": "high"}, ""},
		{"azure-responses", "deployment", "medium", nil, map[string]any{"reasoning.effort": "medium"}, ""},
		{"compat", "model", "low", map[string]any{"reasoningEffort": "high"}, map[string]any{"reasoning_effort": "high"}, ""},
		{"compat", "model", "invalid", nil, map[string]any{"reasoning_effort": nil}, "unsupported"},
		{"anthropic", "claude-sonnet-4-6", "minimal", nil, map[string]any{"thinking.type": "adaptive", "output_config.effort": "low"}, "compatibility"},
		{"anthropic", "claude-sonnet-4-6", "xhigh", nil, map[string]any{"thinking.type": "adaptive", "output_config.effort": "max"}, "compatibility"},
		{"anthropic", "claude-opus-4-7", "xhigh", nil, map[string]any{"thinking.type": "adaptive", "output_config.effort": "xhigh"}, ""},
		{"anthropic", "claude-sonnet-4-5", "medium", nil, map[string]any{"thinking.type": "enabled", "thinking.budget_tokens": float64(19200), "output_config": nil}, ""},
		{"anthropic", "claude-sonnet-4-6", "none", nil, map[string]any{"thinking.type": "disabled", "output_config": nil}, ""},
		{"anthropic", "claude-sonnet-4-6", "none", map[string]any{"effort": "high"}, map[string]any{"thinking": nil, "output_config.effort": "high"}, ""},
		{"anthropic", "claude-sonnet-4-6", "high", map[string]any{"thinking": map[string]any{"type": "disabled"}}, map[string]any{"thinking.type": "disabled", "output_config": nil}, ""},
		{"anthropic", "claude-sonnet-4-6", "high", map[string]any{"thinking": map[string]any{"type": "disabled", "budgetTokens": 3000}}, map[string]any{"thinking.type": "disabled", "thinking.budget_tokens": nil, "output_config": nil}, ""},
		{"anthropic", "claude-sonnet-4-6", "provider-default", nil, map[string]any{"thinking": nil, "output_config": nil}, ""},
		{"anthropic", "claude-sonnet-4-6", "invalid", nil, map[string]any{"thinking": nil, "output_config": nil}, "unsupported"},
		{"google", "gemini-3-pro", "xhigh", nil, map[string]any{"generationConfig.thinkingConfig.thinkingLevel": "high"}, "compatibility"},
		{"google", "gemini-2.5-flash", "medium", nil, map[string]any{"generationConfig.thinkingConfig.thinkingBudget": float64(19661)}, ""},
		{"google", "gemini-2.5-flash", "none", nil, map[string]any{"generationConfig.thinkingConfig.thinkingBudget": float64(0)}, ""},
		{"vertex", "gemini-3.8-flash", "none", nil, map[string]any{"generationConfig.thinkingConfig.thinkingLevel": "low"}, "compatibility"},
		{"vertex", "gemini-3-pro", "high", map[string]any{"thinkingConfig": map[string]any{"thinkingBudget": 0}}, map[string]any{"generationConfig.thinkingConfig.thinkingBudget": float64(0), "generationConfig.thinkingConfig.thinkingLevel": nil}, ""},
		{"xai-chat", "grok-4.5", "xhigh", nil, map[string]any{"reasoning_effort": "high"}, "compatibility"},
		{"xai-chat", "grok-4.20-reasoning", "high", nil, map[string]any{"reasoning_effort": nil}, "unsupported"},
		{"xai-responses", "grok-4.20-reasoning", "high", nil, map[string]any{"reasoning": nil}, "unsupported"},
		{"xai-chat", "grok-4.20-reasoning", "low", map[string]any{"reasoningEffort": "high"}, map[string]any{"reasoning_effort": "high"}, ""},
		{"deepseek", "deepseek-v4-pro", "medium", nil, map[string]any{"reasoning_effort": "high", "thinking.type": "enabled"}, "compatibility"},
		{"deepseek", "deepseek-v4-pro", "xhigh", nil, map[string]any{"reasoning_effort": "max", "thinking.type": "enabled"}, "compatibility"},
		{"deepseek", "deepseek-v4-pro", "none", map[string]any{"reasoningEffort": "high"}, map[string]any{"reasoning_effort": nil, "thinking.type": "disabled"}, ""},
		{"deepseek", "deepseek-v4-pro", "none", map[string]any{"thinking": map[string]any{"type": "enabled"}}, map[string]any{"reasoning_effort": nil, "thinking.type": "enabled"}, ""},
		{"deepseek", "deepseek-v4-pro", "high", map[string]any{"thinking": map[string]any{"type": "disabled"}}, map[string]any{"reasoning_effort": nil, "thinking.type": "disabled"}, ""},
		{"groq", "openai/gpt-oss-120b", "minimal", nil, map[string]any{"reasoning_effort": "low"}, "compatibility"},
		{"groq", "openai/gpt-oss-120b", "none", nil, map[string]any{"reasoning_effort": nil}, "unsupported"},
		{"groq", "qwen/qwen3.6-27b", "none", nil, map[string]any{"reasoning_effort": "none"}, ""},
		{"mistral", "mistral-small-latest", "low", nil, map[string]any{"reasoning_effort": "high"}, "compatibility"},
		{"mistral", "mistral-small-latest", "none", nil, map[string]any{"reasoning_effort": "none"}, ""},
		{"mistral", "mistral-large-latest", "high", nil, map[string]any{"reasoning_effort": nil}, "unsupported"},
		{"cerebras", "model", "xhigh", nil, map[string]any{"reasoning_effort": "high"}, "compatibility"},
		{"cerebras", "model", "none", nil, map[string]any{"reasoning_effort": "none"}, ""},
		{"fireworks", "model", "xhigh", nil, map[string]any{"reasoning_effort": "high"}, ""},
		{"cohere", "command-a-reasoning", "high", nil, map[string]any{"thinking.type": "enabled", "thinking.token_budget": float64(19661)}, ""},
		{"cohere", "command-a-reasoning", "none", nil, map[string]any{"thinking.type": "disabled", "thinking.token_budget": nil}, ""},
		{"cohere", "command-a-reasoning", "high", map[string]any{"thinking": map[string]any{"type": "disabled"}}, map[string]any{"thinking.type": "disabled"}, ""},
	} {
		t.Run(tc.provider+"/"+tc.model+"/"+tc.effort, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "text/event-stream")
			}))
			defer server.Close()
			options := &stream.CallOptions{Messages: []message.Message{message.NewUserMessage("hi")}, Reasoning: tc.effort, ProviderOptions: tc.explicit}
			before, err := json.Marshal(options)
			if err != nil {
				t.Fatal(err)
			}
			events, err := models[tc.provider](server.URL, tc.model).Stream(context.Background(), options)
			if err != nil {
				t.Fatal(err)
			}
			var reasoningWarnings []stream.Warning
			for event := range events {
				if event.Type == stream.EventStart {
					for _, warning := range event.Data.(stream.StartEvent).Warnings {
						if warning.Feature == "reasoning" {
							reasoningWarnings = append(reasoningWarnings, warning)
						}
					}
				}
			}
			if body == nil {
				t.Fatal("request was not sent")
			}
			for path, want := range tc.want {
				var got any = body
				for _, key := range strings.Split(path, ".") {
					object, _ := got.(map[string]any)
					got = object[key]
				}
				if !reflect.DeepEqual(got, want) {
					t.Errorf("%s = %#v, want %#v; body=%v", path, got, want, body)
				}
			}
			if tc.warning == "" && len(reasoningWarnings) != 0 || tc.warning != "" && (len(reasoningWarnings) != 1 || string(reasoningWarnings[0].Type) != tc.warning) {
				t.Errorf("warnings = %v, want %q", reasoningWarnings, tc.warning)
			}
			after, err := json.Marshal(options)
			if err != nil {
				t.Fatal(err)
			}
			if string(before) != string(after) {
				t.Fatal("call options mutated")
			}
		})
	}
}

func TestMapReasoning(t *testing.T) {
	for _, effort := range []string{"", "provider-default", "none", "minimal", "low", "medium", "high", "xhigh", "invalid"} {
		t.Run(effort, func(t *testing.T) {
			got, warnings := provider.OpenAIReasoning(effort, "")
			want := effort
			if effort == "provider-default" || effort == "invalid" {
				want = ""
			}
			if got != want || (len(warnings) != 0) != (effort == "invalid") {
				t.Fatalf("got %q, %v", got, warnings)
			}
		})
	}
}
