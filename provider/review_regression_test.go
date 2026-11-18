package provider_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	bedrock "github.com/airlockrun/goai/provider/amazon-bedrock"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/provider/deepinfra"
	"github.com/airlockrun/goai/provider/deepseek"
	"github.com/airlockrun/goai/provider/fireworks"
	"github.com/airlockrun/goai/provider/groq"
	"github.com/airlockrun/goai/provider/mistral"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/provider/perplexity"
	"github.com/airlockrun/goai/provider/togetherai"
	"github.com/airlockrun/goai/provider/vertexanthropic"
	"github.com/airlockrun/goai/provider/vertexmaas"
	"github.com/airlockrun/goai/provider/xai"
	"github.com/airlockrun/goai/stream"
)

func requireMessageValidator(t *testing.T, model stream.Model) provider.MessageValidator {
	t.Helper()
	validator, ok := model.(provider.MessageValidator)
	if !ok {
		t.Fatal("actual adapter has no complete-message contract")
	}
	return validator
}

func TestReviewNonToolResultPlacement(t *testing.T) {
	for _, adapter := range fileAdapters() {
		if adapter.prompt {
			continue
		}
		t.Run(adapter.name, func(t *testing.T) {
			for _, role := range []message.Role{message.RoleAssistant, message.RoleUser, message.RoleSystem} {
				t.Run(string(role), func(t *testing.T) {
					for _, output := range []struct {
						name  string
						value message.ToolResultOutput
					}{
						{"text", message.TextOutput{Value: "PRIVATE-text"}},
						{"json", message.JSONOutput{Value: map[string]any{"value": "PRIVATE"}}},
						{"media", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "file-data", MediaType: "image/png", Data: "UFJJVkFURQ=="}, {Type: "text", Text: "after"}}}},
					} {
						t.Run(output.name, func(t *testing.T) {
							model := adapter.make()
							messages := []message.Message{{Role: role, Content: message.Content{Parts: []message.Part{message.TextPart{Text: "answer"}, message.ToolResultPart{ToolCallID: "PRIVATE-call", ToolName: "PRIVATE-tool", ProviderExecuted: true, Output: output.value}}}}}
							if err := requireMessageValidator(t, model).ValidateMessages(messages); err == nil || strings.Contains(err.Error(), "PRIVATE") {
								t.Fatalf("unsupported placement admitted or leaked: %v", err)
							}
							capture, failure := captureFileRequest(t, model, messages)
							if capture.requests != 0 || failure == nil || strings.Contains(failure.Error(), "PRIVATE") {
								t.Fatalf("unsupported result lost/textified instead of static pre-HTTP rejection: %v", failure)
							}
						})
					}
				})
			}
		})
	}
}

func TestReviewHostedAnthropicReplay(t *testing.T) {
	for _, adapter := range []struct {
		name string
		make func() stream.Model
	}{
		{"direct", func() stream.Model {
			return anthropic.New(anthropic.Options{APIKey: "fake"}).Model("claude-sonnet-4-6")
		}},
		{"vertex", func() stream.Model {
			return vertexanthropic.New(vertexanthropic.Options{AccessToken: "fake"}).Model("claude-sonnet-4-6")
		}},
		{"bedrock", func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("anthropic.claude-sonnet-4-6")
		}},
	} {
		t.Run(adapter.name, func(t *testing.T) {
			for _, rawJSON := range []bool{false, true} {
				t.Run(map[bool]string{false: "map-metadata", true: "raw-json-metadata"}[rawJSON], func(t *testing.T) {
					raw := map[string]any{"type": "web_search_tool_result", "tool_use_id": "hosted-call", "content": []any{map[string]any{"type": "web_search_result", "url": "https://example.invalid", "title": "hosted", "encrypted_content": "opaque"}}}
					var block any = raw
					if rawJSON {
						bytes, _ := json.Marshal(raw)
						block = json.RawMessage(bytes)
					}
					result := message.ToolResultPart{ToolCallID: "hosted-call", ToolName: "web_search", ProviderExecuted: true, Output: message.JSONOutput{Value: raw["content"]}, ProviderOptions: map[string]any{"anthropic": map[string]any{"rawBlock": block}}}
					messages := []message.Message{message.NewAssistantMessageWithParts(message.ToolCallPart{ID: "hosted-call", Name: "web_search", Input: json.RawMessage(`{}`), ProviderExecuted: true}, result, message.TextPart{Text: "answer"})}
					model := adapter.make()
					if err := requireMessageValidator(t, model).ValidateMessages(messages); err != nil {
						t.Fatalf("legitimate native hosted replay rejected: %v", err)
					}
					capture, failure := captureFileRequest(t, model, messages)
					if capture.requests != 1 {
						t.Fatalf("native hosted replay did not serialize: %v", failure)
					}
					var body map[string]any
					if err := json.Unmarshal(capture.body, &body); err != nil {
						t.Fatal(err)
					}
					parts := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
					if len(parts) != 3 || parts[0].(map[string]any)["type"] != "server_tool_use" || parts[1].(map[string]any)["type"] != "web_search_tool_result" || parts[1].(map[string]any)["tool_use_id"] != "hosted-call" {
						t.Fatal("native replay order/correlation lost")
					}
					// Generic media cannot hide behind native replay metadata.
					result.Output = message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "application/pdf", Data: "UFJJVkFURQ=="}}}
					bad := []message.Message{message.NewAssistantMessageWithParts(result)}
					if err := requireMessageValidator(t, model).ValidateMessages(bad); err == nil {
						t.Fatal("generic assistant media admitted via raw replay metadata")
					}
					capture, failure = captureFileRequest(t, model, bad)
					if capture.requests != 0 || failure == nil {
						t.Fatal("generic assistant media reached HTTP")
					}
				})
			}
		})
	}
}

func TestReviewGeminiSiblingTextMediaCorrelation(t *testing.T) {
	for _, adapter := range fileAdapters() {
		if adapter.shape != "gemini-native" && adapter.shape != "gemini-sibling" {
			continue
		}
		t.Run(adapter.name, func(t *testing.T) {
			for _, multipart := range []bool{false, true} {
				t.Run(map[bool]string{false: "scalar-json-final", true: "multipart-final"}[multipart], func(t *testing.T) {
					var output message.ToolResultOutput = message.JSONOutput{Value: map[string]any{"value": 42}}
					wantText := message.ToolOutputWire(output) + "LEFTRIGHT"
					wantMedia := []string{"U0VDT05E", "VEhJUkQ="}
					if multipart {
						output = message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "final-before"}, {Type: "file-data", MediaType: "image/png", Data: "RklSU1Q="}, {Type: "text", Text: "final-after"}}}
						wantText = "final-beforefinal-afterLEFTRIGHT"
						wantMedia = append([]string{"RklSU1Q="}, wantMedia...)
					}
					msg := message.NewToolMessage("first-call", "first-tool", message.TextOutput{Value: "first"})
					msg.Content.Parts = append(msg.Content.Parts, message.ToolResultPart{ToolCallID: "last-call", ToolName: "last-tool", Output: output}, message.TextPart{Text: "LEFT"}, message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "U0VDT05E"}}, message.TextPart{Text: "RIGHT"}, message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "VEhJUkQ="}})
					model := adapter.make()
					if err := requireMessageValidator(t, model).ValidateMessages([]message.Message{msg}); err != nil {
						t.Fatal(err)
					}
					capture, failure := captureFileRequest(t, model, []message.Message{msg})
					if capture.requests != 1 {
						t.Fatal(failure)
					}
					var body map[string]any
					if err := json.Unmarshal(capture.body, &body); err != nil {
						t.Fatal(err)
					}
					parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
					responses := 0
					var media []any
					for _, part := range parts {
						part := part.(map[string]any)
						if response, ok := part["functionResponse"].(map[string]any); ok {
							responses++
							if responses == 1 {
								if response["id"] != "first-call" || response["parts"] != nil {
									t.Fatal("first result stole attachments")
								}
							} else {
								if response["id"] != "last-call" || response["name"] != "last-tool" || response["response"].(map[string]any)["result"] != wantText {
									t.Fatal("final real result correlation/text changed")
								}
								if adapter.shape == "gemini-native" {
									media = response["parts"].([]any)
								}
							}
						} else if adapter.shape == "gemini-sibling" {
							media = append(media, part)
						} else {
							t.Fatal("unexpected native sibling part")
						}
					}
					if responses != 2 || len(media) != len(wantMedia) {
						t.Fatal("duplicate result or lost media")
					}
					for i, want := range wantMedia {
						if media[i].(map[string]any)["inlineData"].(map[string]any)["data"] != want {
							t.Fatal("media order changed")
						}
					}
				})
			}
		})
	}
}

func TestReviewDeepSeekSiblingAdmissionAndWire(t *testing.T) {
	for _, urlSource := range []bool{false, true} {
		t.Run(map[bool]string{false: "inline", true: "url"}[urlSource], func(t *testing.T) {
			model := deepseek.New(deepseek.Options{APIKey: "fake"}).Model("deepseek-v4")
			file := message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "c3BlY2ltZW4="}}
			want := "data:image/png;base64,c3BlY2ltZW4="
			if urlSource {
				want = "https://example.invalid/image.png"
				file.Data = message.FileDataURL{URL: want}
			}
			msg := message.NewToolMessage("first-call", "first", message.TextOutput{Value: "one"})
			msg.Content.Parts = append(msg.Content.Parts, message.ToolResultPart{ToolCallID: "last-call", ToolName: "last", Output: message.JSONOutput{Value: map[string]any{"answer": 42}}}, message.TextPart{Text: "LEFT"}, file, message.TextPart{Text: "RIGHT"})
			messages := []message.Message{msg}
			if err := provider.ValidateMessageFiles(messages, model.(provider.FileValidator)); err != nil {
				t.Fatal(err)
			}
			if err := requireMessageValidator(t, model).ValidateMessages(messages); err != nil {
				t.Fatal(err)
			}
			capture, failure := captureFileRequest(t, model, messages)
			if capture.requests != 1 {
				t.Fatalf("admission/serializer mismatch: %v", failure)
			}
			var body map[string]any
			if err := json.Unmarshal(capture.body, &body); err != nil {
				t.Fatal(err)
			}
			wire := body["messages"].([]any)
			if len(wire) != 2 || wire[0].(map[string]any)["tool_call_id"] != "first-call" || wire[0].(map[string]any)["content"] != "one" {
				t.Fatal("first result changed")
			}
			last := wire[1].(map[string]any)
			parts := last["content"].([]any)
			if last["tool_call_id"] != "last-call" || len(parts) != 4 || parts[0].(map[string]any)["text"] != `{"answer":42}` || parts[1].(map[string]any)["text"] != "LEFT" || parts[2].(map[string]any)["image_url"].(map[string]any)["url"] != want || parts[3].(map[string]any)["text"] != "RIGHT" {
				t.Fatal("final result typed attachment order/correlation changed")
			}
		})
	}
}

func TestReviewUnknownCompatibleEndpointAdmission(t *testing.T) {
	for _, identity := range []string{"unknown", "mistral", "deepseek", "xai"} {
		t.Run(identity, func(t *testing.T) {
			for _, mime := range []string{"image/png", "video/mp4"} {
				t.Run(mime, func(t *testing.T) {
					model := openaicompat.New(openaicompat.Options{ProviderID: identity, APIKey: "fake", BaseURL: "https://unknown.invalid"}).Model("unknown")
					file := message.FilePart{MimeType: mime, Data: message.FileDataBytes{Data: "UFJJVkFURQ=="}}
					if err := model.(provider.FileValidator).ValidateFile(message.RoleUser, file); err == nil {
						t.Fatal("vendor string inferred endpoint support")
					}
					messages := []message.Message{message.NewUserMessageWithParts(file)}
					if err := requireMessageValidator(t, model).ValidateMessages(messages); err == nil {
						t.Fatal("unknown binary complete-message admission succeeded")
					}
					capture, failure := captureFileRequest(t, model, messages)
					if capture.requests != 0 || failure == nil || strings.Contains(failure.Error(), "UFJJVkFURQ") {
						t.Fatal("unknown binary request was admitted or error leaked")
					}
				})
			}
			t.Run("ordinary-text", func(t *testing.T) {
				model := openaicompat.New(openaicompat.Options{ProviderID: identity, BaseURL: "https://unknown.invalid"}).Model("unknown")
				messages := []message.Message{message.NewUserMessage("ordinary-text"), message.NewToolMessage("call", "tool", message.JSONOutput{Value: map[string]any{"value": 42}})}
				if err := requireMessageValidator(t, model).ValidateMessages(messages); err != nil {
					t.Fatal(err)
				}
				capture, failure := captureFileRequest(t, model, messages)
				if capture.requests != 1 {
					t.Fatal(failure)
				}
			})
		})
	}
}

func TestReviewExplicitCompatibleUserDeclaration(t *testing.T) {
	for _, mime := range []string{"image/png", "video/mp4"} {
		t.Run(mime, func(t *testing.T) {
			options := openaicompat.Options{BaseURL: "https://declared.invalid", FileValidator: func(model string, role message.Role, file message.FilePart) error {
				if model != "declared-model" || role != message.RoleUser {
					return provider.ErrUnsupportedFile
				}
				return provider.ValidateInlineOrURLFile(file, []string{mime}, false)
			}}
			file := message.FilePart{MimeType: mime, Data: message.FileDataBytes{Data: "c3BlY2ltZW4="}}
			messages := []message.Message{message.NewUserMessageWithParts(file)}
			for _, modelID := range []string{"declared-model", "other-model"} {
				t.Run(modelID, func(t *testing.T) {
					model := openaicompat.New(options).Model(modelID)
					failure := requireMessageValidator(t, model).ValidateMessages(messages)
					if (failure == nil) != (modelID == "declared-model") {
						t.Fatal("explicit model gate mismatch")
					}
					capture, failure := captureFileRequest(t, model, messages)
					if modelID != "declared-model" {
						if capture.requests != 0 || failure == nil {
							t.Fatal("undeclared model admitted")
						}
						return
					}
					if capture.requests != 1 {
						t.Fatal(failure)
					}
					var body map[string]any
					if err := json.Unmarshal(capture.body, &body); err != nil {
						t.Fatal(err)
					}
					part := body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
					want := "image_url"
					if mime == "video/mp4" {
						want = "video_url"
					}
					if part["type"] != want || part[want].(map[string]any)["url"] != "data:"+mime+";base64,c3BlY2ltZW4=" {
						t.Fatal("explicit default serializer declaration did not use typed media")
					}
				})
			}
		})
	}
}

func TestReviewKnownWrapperImageDeclarations(t *testing.T) {
	for _, adapter := range []struct {
		name string
		make func() stream.Model
	}{
		{"mistral", func() stream.Model { return mistral.New(mistral.Options{APIKey: "fake"}).Model("pixtral-large-latest") }},
		{"groq", func() stream.Model { return groq.New(groq.Options{APIKey: "fake"}).Model("vision") }},
		{"fireworks", func() stream.Model { return fireworks.New(fireworks.Options{APIKey: "fake"}).Model("vision") }},
		{"together", func() stream.Model { return togetherai.New(togetherai.Options{APIKey: "fake"}).Model("vision") }},
		{"deepinfra", func() stream.Model { return deepinfra.New(deepinfra.Options{APIKey: "fake"}).Model("vision") }},
		{"perplexity", func() stream.Model { return perplexity.New(perplexity.Options{APIKey: "fake"}).Model("vision") }},
		{"xai-chat", func() stream.Model { return xai.New(xai.Options{APIKey: "fake"}).Chat("grok-4") }},
		{"vertex-maas", func() stream.Model { return vertexmaas.New(vertexmaas.Options{AccessToken: "fake"}).Model("vision") }},
	} {
		t.Run(adapter.name, func(t *testing.T) {
			for _, mime := range []string{"image/png", "video/mp4"} {
				t.Run(mime, func(t *testing.T) {
					model := adapter.make()
					messages := []message.Message{message.NewUserMessageWithParts(message.FilePart{MimeType: mime, Data: message.FileDataBytes{Data: "c3BlY2ltZW4="}})}
					admitted := requireMessageValidator(t, model).ValidateMessages(messages) == nil
					if admitted != (mime == "image/png") {
						t.Fatal("known wrapper contract mismatch")
					}
					capture, failure := captureFileRequest(t, model, messages)
					if mime == "image/png" {
						if capture.requests != 1 || !strings.Contains(string(capture.body), `"type":"image_url"`) {
							t.Fatalf("known typed image transport lost: %v", failure)
						}
					} else if capture.requests != 0 || failure == nil {
						t.Fatal("undeclared wrapper video inferred supported")
					}
				})
			}
		})
	}
}
