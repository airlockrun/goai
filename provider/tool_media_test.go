package provider_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	bedrock "github.com/airlockrun/goai/provider/amazon-bedrock"
	"github.com/airlockrun/goai/provider/anthropic"
	"github.com/airlockrun/goai/provider/azure"
	"github.com/airlockrun/goai/provider/baseten"
	"github.com/airlockrun/goai/provider/cerebras"
	"github.com/airlockrun/goai/provider/cohere"
	"github.com/airlockrun/goai/provider/deepseek"
	"github.com/airlockrun/goai/provider/google"
	"github.com/airlockrun/goai/provider/huggingface"
	"github.com/airlockrun/goai/provider/mistral"
	"github.com/airlockrun/goai/provider/openai"
	"github.com/airlockrun/goai/provider/openaicompat"
	"github.com/airlockrun/goai/provider/openresponses"
	"github.com/airlockrun/goai/provider/vertex"
	"github.com/airlockrun/goai/provider/vertexanthropic"
	"github.com/airlockrun/goai/provider/xai"
	"github.com/airlockrun/goai/stream"
)

type fileCapture struct {
	body     []byte
	requests int
}

func (c *fileCapture) RoundTrip(request *http.Request) (*http.Response, error) {
	c.requests++
	var err error
	c.body, err = io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: http.StatusBadRequest, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":"owned fake transport"}`)), Request: request}, nil
}

// Every request terminates in this transport; these tests never open sockets,
// load real credentials, or claim remote model acceptance.
func captureFileRequest(t *testing.T, model stream.Model, messages []message.Message) (*fileCapture, error) {
	t.Helper()
	capture := &fileCapture{}
	old := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: capture}
	t.Cleanup(func() { http.DefaultClient = old })
	before, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	events, failure := model.Stream(context.Background(), &stream.CallOptions{Messages: messages})
	if events != nil {
		for event := range events {
			if value, ok := event.Data.(stream.ErrorEvent); ok {
				failure = value.Error
			}
		}
	}
	after, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("adapter mutated message history")
	}
	return capture, failure
}

type fileAdapter struct {
	name, shape        string
	prompt, image, pdf bool
	make               func() stream.Model
}

func fileAdapters() []fileAdapter {
	return []fileAdapter{
		{"openai-responses", "responses", false, true, true, func() stream.Model { return openai.New(provider.Options{APIKey: "fake"}).Model("gpt-4.1") }},
		{"azure-responses", "responses", false, true, true, func() stream.Model {
			return azure.New(azure.Options{APIKey: "fake", ResourceName: "fake"}).Model("gpt-4.1")
		}},
		{"openresponses", "responses", false, true, true, func() stream.Model {
			return openresponses.New(openresponses.Options{APIKey: "fake", BaseURL: "https://fake.invalid"}).Model("vision")
		}},
		{"huggingface-responses", "responses", false, true, true, func() stream.Model { return huggingface.New(huggingface.Options{APIKey: "fake"}).Model("vision") }},
		{"openai-chat", "chat", false, false, false, func() stream.Model { return openai.New(provider.Options{APIKey: "fake"}).Chat("gpt-4.1") }},
		{"azure-chat", "chat", false, false, false, func() stream.Model {
			return azure.New(azure.Options{APIKey: "fake", ResourceName: "fake"}).Chat("gpt-4.1")
		}},
		{"compatible", "chat", false, false, false, func() stream.Model {
			return openaicompat.New(openaicompat.Options{APIKey: "fake", BaseURL: "https://fake.invalid"}).Model("unknown")
		}},
		{"cerebras-custom-converter", "chat", false, false, false, func() stream.Model { return cerebras.New(cerebras.Options{APIKey: "fake"}).Model("text") }},
		{"cohere-native", "chat", false, false, false, func() stream.Model { return cohere.New(cohere.Options{APIKey: "fake"}).Model("command-a-vision") }},
		{"mistral-custom-converter", "chat", false, false, false, func() stream.Model { return mistral.New(mistral.Options{APIKey: "fake"}).Model("mistral-large-latest") }},
		{"deepseek", "deepseek", false, true, false, func() stream.Model { return deepseek.New(deepseek.Options{APIKey: "fake"}).Model("deepseek-v4") }},
		{"anthropic", "anthropic", false, true, true, func() stream.Model {
			return anthropic.New(anthropic.Options{APIKey: "fake"}).Model("claude-sonnet-4-6")
		}},
		{"vertex-anthropic", "anthropic", false, true, true, func() stream.Model {
			return vertexanthropic.New(vertexanthropic.Options{AccessToken: "fake", Project: "fake", Location: "fake"}).Model("claude-sonnet-4-6")
		}},
		{"bedrock-anthropic", "anthropic", false, true, true, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("anthropic.claude-sonnet-4-6")
		}},
		{"google-3", "gemini-native", false, true, true, func() stream.Model { return google.New(google.Options{APIKey: "fake"}).Model("gemini-3-flash-preview") }},
		{"google-2.5", "gemini-sibling", false, true, true, func() stream.Model { return google.New(google.Options{APIKey: "fake"}).Model("gemini-2.5-flash") }},
		{"vertex-3", "gemini-native", false, true, true, func() stream.Model {
			return vertex.New(vertex.Options{AccessToken: "fake"}).Model("gemini-3-pro-preview")
		}},
		{"vertex-2.5", "gemini-sibling", false, true, true, func() stream.Model { return vertex.New(vertex.Options{AccessToken: "fake"}).Model("gemini-2.5-flash") }},
		{"xai-responses", "responses", false, true, true, func() stream.Model { return xai.New(xai.Options{APIKey: "fake"}).Model("grok-4") }},
		{"xai-chat", "chat", false, false, false, func() stream.Model { return xai.New(xai.Options{APIKey: "fake"}).Chat("grok-4") }},
		{"bedrock-nova-pro", "converse", false, true, true, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("amazon.nova-pro-v1:0")
		}},
		{"bedrock-nova-micro", "converse", false, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("amazon.nova-micro-v1:0")
		}},
		{"bedrock-unknown", "converse", false, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("unknown.model")
		}},
		{"bedrock-titan", "prompt", true, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("amazon.titan-text")
		}},
		{"bedrock-llama", "prompt", true, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("meta.llama")
		}},
		{"bedrock-mistral", "prompt", true, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("mistral.test")
		}},
		{"bedrock-cohere", "prompt", true, false, false, func() stream.Model {
			return bedrock.New(bedrock.Options{AccessKeyID: "fake", SecretAccessKey: "fake"}).Model("cohere.test")
		}},
		{"baseten", "prompt", true, false, false, func() stream.Model { return baseten.New(baseten.Options{APIKey: "fake"}).Model("text") }},
		{"huggingface-textgeneration", "prompt", true, false, false, func() stream.Model {
			return huggingface.New(huggingface.Options{APIKey: "fake"}).TextGeneration("text")
		}},
	}
}

func hasStringLeaf(value any, want string) bool {
	switch value := value.(type) {
	case string:
		return value == want
	case []any:
		for _, item := range value {
			if hasStringLeaf(item, want) {
				return true
			}
		}
	case map[string]any:
		for _, item := range value {
			if hasStringLeaf(item, want) {
				return true
			}
		}
	}
	return false
}

func TestToolMediaHTTPMatrix(t *testing.T) {
	const payload = "TUVESUFTUEVDSU1FTg=="
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			for _, variant := range []struct {
				name   string
				output message.ToolResultOutput
				want   bool
				media  bool
			}{
				{"text", message.TextOutput{Value: "plain-tool-text"}, !adapter.prompt, false},
				{"json", message.JSONOutput{Value: map[string]any{"answer": 42}}, !adapter.prompt, false},
				{"text-content", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "first"}, {Type: "text", Text: "last"}}}, !adapter.prompt, false},
				{"image", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "image-data", MediaType: "image/png", Data: payload}, {Type: "text", Text: "after"}}}, adapter.image, true},
				{"generic-image-file", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "file-data", MediaType: "image/png", Data: payload}, {Type: "text", Text: "after"}}}, adapter.image, true},
				{"pdf", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "file-data", MediaType: "application/pdf", Data: payload, Filename: "report.pdf"}, {Type: "text", Text: "after"}}}, adapter.pdf, true},
				{"audio", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "audio/wav", Data: payload}}}, adapter.shape == "gemini-native", true},
				{"video", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "video/mp4", Data: payload}}}, adapter.shape == "gemini-native", true},
				{"binary", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "application/octet-stream", Data: payload}}}, false, true},
				{"custom-mixed", message.ContentOutput{Value: []message.ToolContentItem{{Type: "text", Text: "before"}, {Type: "private-custom-MIME", Data: payload}}}, false, true},
				{"reference", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-reference", ProviderReference: map[string]string{"provider": payload}}}}, false, true},
			} {
				t.Run(variant.name, func(t *testing.T) {
					capture, failure := captureFileRequest(t, adapter.make(), []message.Message{message.NewToolMessage("private-call", "private-tool", variant.output)})
					if !variant.want {
						if failure == nil || capture.requests != 0 {
							t.Fatalf("expected pre-HTTP rejection, requests=%d error=%v", capture.requests, failure)
						}
						for _, secret := range []string{payload, "private-call", "private-tool", "private-custom-MIME", "audio/wav", "video/mp4"} {
							if strings.Contains(failure.Error(), secret) {
								t.Fatal("validation diagnostic leaks private input")
							}
						}
						return
					}
					if capture.requests != 1 {
						t.Fatalf("expected typed request, error=%v", failure)
					}
					var body any
					if err := json.Unmarshal(capture.body, &body); err != nil {
						t.Fatal(err)
					}
					if variant.name == "text" || variant.name == "json" {
						if !hasStringLeaf(body, message.ToolOutputWire(variant.output)) {
							t.Fatal("scalar output not preserved")
						}
					}
					if variant.media {
						if !strings.Contains(string(capture.body), payload) || strings.Contains(string(capture.body), `\"data\"`) {
							t.Fatal("media missing or textified")
						}
						if !hasStringLeaf(body, "private-call") {
							t.Fatal("tool correlation lost")
						}
						assertToolWireShape(t, adapter.shape, body, variant.name)
					}
				})
			}
		})
	}
}

func assertToolWireShape(t *testing.T, shape string, value any, media string) {
	t.Helper()
	body := value.(map[string]any)
	switch shape {
	case "responses":
		input := body["input"].([]any)
		output := input[len(input)-1].(map[string]any)["output"].([]any)
		want := "input_image"
		if media == "pdf" {
			want = "input_file"
		}
		if len(output) != 3 || output[1].(map[string]any)["type"] != want {
			t.Fatalf("incorrect response parts: %+v", output)
		}
		if media == "pdf" && body["model"] == "grok-4" {
			file := output[1].(map[string]any)
			if file["file_data"] != "TUVESUFTUEVDSU1FTg==" || file["filename"] != "report.pdf" || file["mime_type"] != "application/pdf" {
				t.Fatal("xAI inline file contract mismatch")
			}
		}
	case "gemini-native", "gemini-sibling":
		contents := body["contents"].([]any)
		parts := contents[len(contents)-1].(map[string]any)["parts"].([]any)
		response := parts[0].(map[string]any)["functionResponse"].(map[string]any)
		if response["id"] != "private-call" {
			t.Fatal("Gemini response ID missing")
		}
		if shape == "gemini-native" {
			if len(parts) != 1 || response["parts"] == nil {
				t.Fatal("native media is not nested in functionResponse")
			}
			mediaParts := response["parts"].([]any)
			if len(mediaParts) != 1 {
				t.Fatal("native media order or count changed")
			}
			inline, ok := mediaParts[0].(map[string]any)["inlineData"].(map[string]any)
			mime := map[string]string{"image": "image/png", "generic-image-file": "image/png", "pdf": "application/pdf", "audio": "audio/wav", "video": "video/mp4"}[media]
			if !ok || inline["mimeType"] != mime || inline["data"] != "TUVESUFTUEVDSU1FTg==" {
				t.Fatal("native media MIME/source contract mismatch")
			}
		} else {
			if len(parts) != 2 || response["parts"] != nil || parts[1].(map[string]any)["inlineData"] == nil {
				t.Fatal("older-model sibling media contract mismatch")
			}
		}
	case "anthropic":
		messages := body["messages"].([]any)
		tool := messages[len(messages)-1].(map[string]any)["content"].([]any)[0].(map[string]any)
		parts := tool["content"].([]any)
		want := "image"
		if media == "pdf" {
			want = "document"
		}
		if len(parts) != 3 || parts[1].(map[string]any)["type"] != want {
			t.Fatal("Anthropic tool media contract mismatch")
		}
	case "converse":
		messages := body["messages"].([]any)
		tool := messages[len(messages)-1].(map[string]any)["content"].([]any)[0].(map[string]any)["toolResult"].(map[string]any)
		parts := tool["content"].([]any)
		want := "image"
		if media == "pdf" {
			want = "document"
		}
		if len(parts) != 3 || parts[1].(map[string]any)[want] == nil {
			t.Fatal("Converse native union missing")
		}
	case "deepseek":
		messages := body["messages"].([]any)
		parts := messages[len(messages)-1].(map[string]any)["content"].([]any)
		if len(parts) != 3 || parts[1].(map[string]any)["type"] != "image_url" {
			t.Fatal("DeepSeek typed images missing")
		}
	default:
		t.Fatal("unexpected media-capable transport")
	}
}

func TestFileValidatorContract(t *testing.T) {
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			model := adapter.make()
			validator, ok := model.(provider.FileValidator)
			if adapter.prompt {
				return
			}
			if !ok {
				t.Fatal("file contract not exposed by actual model")
			}
			for _, mime := range []string{"image/png", "application/pdf", "audio/wav", "video/mp4", "application/octet-stream"} {
				file := message.FilePart{MimeType: mime, Data: message.FileDataBytes{Data: "c3BlY2ltZW4="}}
				want := (mime == "image/png" && adapter.image) || (mime == "application/pdf" && adapter.pdf)
				if adapter.shape == "gemini-native" && (mime == "audio/wav" || mime == "video/mp4") {
					want = true
				}
				if got := validator.ValidateFile(message.RoleTool, file) == nil; got != want {
					t.Fatalf("MIME=%s support=%t want=%t", mime, got, want)
				}
				if err := validator.ValidateFile(message.RoleSystem, file); err == nil {
					t.Fatal("system role unexpectedly supports files")
				}
			}
		})
	}
}

func TestAzureChatMultipleToolResults(t *testing.T) {
	first := message.ToolResultPart{ToolCallID: "first", ToolName: "read", Output: message.TextOutput{Value: "one"}}
	second := message.ToolResultPart{ToolCallID: "second", ToolName: "read", Output: message.JSONOutput{Value: map[string]any{"two": 2}}}
	capture, _ := captureFileRequest(t, azure.New(azure.Options{APIKey: "fake", ResourceName: "fake"}).Chat("gpt-4.1"), []message.Message{{Role: message.RoleTool, Content: message.Content{Parts: []message.Part{first, second}}}})
	var body map[string]any
	if err := json.Unmarshal(capture.body, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	if len(messages) != 2 || !reflect.DeepEqual(messages[0], map[string]any{"role": "tool", "tool_call_id": "first", "content": "one"}) || messages[1].(map[string]any)["tool_call_id"] != "second" {
		t.Fatal("multiple tool results lost or reordered")
	}
}

func TestToolFileURLSources(t *testing.T) {
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			for _, kind := range []string{"image-url", "file-url"} {
				t.Run(kind, func(t *testing.T) {
					mime := "image/png"
					if kind == "file-url" {
						mime = "application/pdf"
					}
					url := "https://files.example.invalid/PRIVATE-SIGNED-URL?token=PRIVATE"
					want := adapter.image
					if kind == "file-url" {
						want = adapter.pdf
					}
					if adapter.name == "vertex-anthropic" || adapter.name == "bedrock-anthropic" || strings.HasPrefix(adapter.name, "bedrock-") || strings.HasPrefix(adapter.name, "google-") || strings.HasPrefix(adapter.name, "vertex-") {
						want = false
					}
					capture, failure := captureFileRequest(t, adapter.make(), []message.Message{message.NewToolMessage("private-call", "read", message.ContentOutput{Value: []message.ToolContentItem{{Type: kind, MediaType: mime, URL: url}}})})
					if want {
						if capture.requests != 1 || !strings.Contains(string(capture.body), "PRIVATE-SIGNED-URL") {
							t.Fatalf("URL lost, error=%v", failure)
						}
					} else if failure == nil || capture.requests != 0 || strings.Contains(failure.Error(), "PRIVATE") {
						t.Fatalf("expected static pre-HTTP rejection, requests=%d error=%v", capture.requests, failure)
					}
				})
			}
		})
	}
	for _, vertexAPI := range []bool{false, true} {
		t.Run(map[bool]string{false: "google-files", true: "vertex-gcs"}[vertexAPI], func(t *testing.T) {
			var model stream.Model = google.New(google.Options{APIKey: "fake"}).Model("gemini-3-pro-preview")
			url := "https://generativelanguage.googleapis.com/v1beta/files/report"
			if vertexAPI {
				model = vertex.New(vertex.Options{AccessToken: "fake"}).Model("gemini-3-pro-preview")
				url = "gs://bucket/report.pdf"
			}
			capture, failure := captureFileRequest(t, model, []message.Message{message.NewToolMessage("call", "read", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-url", MediaType: "application/pdf", URL: url}}})})
			if capture.requests != 1 || !strings.Contains(string(capture.body), url) || !strings.Contains(string(capture.body), `"fileData"`) {
				t.Fatalf("native provider URI missing: %v", failure)
			}
		})
	}
}

func TestSiblingToolFilesAndCorrelation(t *testing.T) {
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			msg := message.NewToolMessage("first", "read", message.TextOutput{Value: "one"})
			msg.Content.Parts = append(msg.Content.Parts, message.ToolResultPart{ToolCallID: "second", ToolName: "read", Output: message.TextOutput{Value: "two"}}, message.FilePart{MimeType: "image/png", Data: message.FileDataBytes{Data: "U0lCTElORw=="}})
			capture, failure := captureFileRequest(t, adapter.make(), []message.Message{msg})
			want := adapter.image
			if !want {
				if failure == nil || capture.requests != 0 {
					t.Fatalf("expected pre-HTTP rejection: %v", failure)
				}
				return
			}
			if capture.requests != 1 || strings.Count(string(capture.body), "U0lCTElORw==") != 1 {
				t.Fatalf("sibling file lost or duplicated: %v", failure)
			}
			var body any
			if err := json.Unmarshal(capture.body, &body); err != nil {
				t.Fatal(err)
			}
			if !hasStringLeaf(body, "first") || !hasStringLeaf(body, "second") {
				t.Fatal("tool correlation lost")
			}
			if adapter.shape == "gemini-native" {
				parts := body.(map[string]any)["contents"].([]any)[0].(map[string]any)["parts"].([]any)
				if len(parts) != 2 || parts[0].(map[string]any)["functionResponse"].(map[string]any)["parts"] != nil || parts[1].(map[string]any)["functionResponse"].(map[string]any)["parts"] == nil {
					t.Fatal("sibling file must attach only to final native response")
				}
			}
		})
	}
}

func TestUserFileRoleContracts(t *testing.T) {
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			for _, mime := range []string{"image/png", "application/pdf", "audio/wav", "video/mp4", "application/octet-stream"} {
				t.Run(mime, func(t *testing.T) {
					model := adapter.make()
					file := message.FilePart{MimeType: mime, Data: message.FileDataBytes{Data: "Uk9MRVNQRUNJTUVOVA=="}, Filename: "report.pdf"}
					validator, ok := model.(provider.FileValidator)
					want := false
					switch mime {
					case "image/png":
						want = !adapter.prompt && adapter.name != "bedrock-nova-micro" && adapter.name != "bedrock-unknown" && adapter.name != "compatible" && adapter.name != "cerebras-custom-converter"
					case "application/pdf":
						want = adapter.pdf || adapter.name == "openai-chat"
					case "audio/wav":
						want = strings.HasPrefix(adapter.name, "google-") || adapter.name == "vertex-3" || adapter.name == "vertex-2.5" || adapter.name == "openai-chat"
					case "video/mp4":
						want = strings.HasPrefix(adapter.name, "google-") || adapter.name == "vertex-3" || adapter.name == "vertex-2.5"
					}
					if ok && (validator.ValidateFile(message.RoleUser, file) == nil) != want {
						t.Fatal("declared user-role contract mismatch")
					}
					if !ok && want {
						t.Fatal("supported adapter has no file contract")
					}
					capture, failure := captureFileRequest(t, model, []message.Message{message.NewUserMessageWithParts(file)})
					if !want {
						if failure == nil || capture.requests != 0 {
							t.Fatalf("unsupported user file dropped instead of rejected: %v", failure)
						}
						return
					}
					if capture.requests != 1 || !strings.Contains(string(capture.body), "Uk9MRVNQRUNJTUVOVA==") {
						t.Fatalf("supported user file dropped: %v", failure)
					}
				})
			}
		})
	}
}

func TestPromptAdaptersRetainPlainTextAndRejectMultipart(t *testing.T) {
	for _, adapter := range fileAdapters() {
		if !adapter.prompt {
			continue
		}
		t.Run(adapter.name, func(t *testing.T) {
			t.Run("plain", func(t *testing.T) {
				capture, failure := captureFileRequest(t, adapter.make(), []message.Message{message.NewUserMessage("ordinary-text-control")})
				if capture.requests != 1 || !strings.Contains(string(capture.body), "ordinary-text-control") {
					t.Fatalf("plain text broken: %v", failure)
				}
			})
			t.Run("multipart-text", func(t *testing.T) {
				capture, failure := captureFileRequest(t, adapter.make(), []message.Message{message.NewUserMessageWithParts(message.TextPart{Text: "must-not-drop"})})
				if failure == nil || capture.requests != 0 {
					t.Fatal("unrepresentable multipart text silently dropped")
				}
			})
		})
	}
}

func TestUnknownModelsAndSourcesRejectBeforeHTTP(t *testing.T) {
	for _, model := range []stream.Model{
		google.New(google.Options{APIKey: "fake"}).Model("unknown-gemini"),
		vertex.New(vertex.Options{AccessToken: "fake"}).Model("unknown-gemini"),
		xai.New(xai.Options{APIKey: "fake"}).Model("grok-3"),
	} {
		t.Run(model.Provider(), func(t *testing.T) {
			capture, failure := captureFileRequest(t, model, []message.Message{message.NewToolMessage("call", "read", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "image/png", Data: "c3BlY2ltZW4="}}})})
			if failure == nil || capture.requests != 0 {
				t.Fatal("unknown model inferred file-capable")
			}
		})
	}
	for _, adapter := range fileAdapters() {
		t.Run(adapter.name, func(t *testing.T) {
			for _, data := range []message.FileData{message.FileDataReference{Reference: map[string]any{"PRIVATE": "PRIVATE"}}, message.FileDataURL{URL: "airlock-file:PRIVATE"}, message.FileDataURL{URL: "https://private:PRIVATE@example.invalid/file"}} {
				msg := message.NewToolMessage("PRIVATE-call", "PRIVATE-name", message.TextOutput{Value: "summary"})
				msg.Content.Parts = append(msg.Content.Parts, message.FilePart{MimeType: "application/pdf", Data: data})
				capture, failure := captureFileRequest(t, adapter.make(), []message.Message{msg})
				if failure == nil || capture.requests != 0 || strings.Contains(failure.Error(), "PRIVATE") {
					t.Fatal("unsupported source not rejected statically")
				}
			}
		})
	}
}

func TestAnthropicImageURLWithoutMIME(t *testing.T) {
	capture, failure := captureFileRequest(t, anthropic.New(anthropic.Options{APIKey: "fake"}).Model("claude-sonnet-4-6"), []message.Message{message.NewToolMessage("call", "read", message.ContentOutput{Value: []message.ToolContentItem{{Type: "image-url", URL: "https://example.invalid/image.png"}}})})
	if capture.requests != 1 {
		t.Fatalf("image URL rejected: %v", failure)
	}
	var body map[string]any
	if err := json.Unmarshal(capture.body, &body); err != nil {
		t.Fatal(err)
	}
	block := body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if block["type"] != "image" || block["source"].(map[string]any)["type"] != "url" || block["source"].(map[string]any)["url"] != "https://example.invalid/image.png" {
		t.Fatal("image URL silently lost or mis-typed")
	}
}

func TestCompatibleToolFilesRequireDeclaredSerializer(t *testing.T) {
	for _, test := range []struct {
		name                 string
		validator, converter bool
		want                 bool
	}{
		{"unknown", false, false, false},
		{"serializer-only", false, true, false},
		{"validator-only", true, false, false},
		{"declared-pair", true, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := openaicompat.Options{APIKey: "fake", BaseURL: "https://fake.invalid"}
			if test.validator {
				opts.FileValidator = func(_ string, role message.Role, file message.FilePart) error {
					if role != message.RoleTool {
						return provider.ErrUnsupportedFile
					}
					return provider.ValidateInlineOrURLFile(file, []string{"image/png"}, false)
				}
			}
			if test.converter {
				opts.MessageConverter = func(_ string, messages []message.Message) ([]any, error) {
					result := messages[0].Content.Parts[0].(message.ToolResultPart)
					return []any{map[string]any{"role": "tool", "tool_call_id": result.ToolCallID, "content": []any{map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,c3BlY2ltZW4="}}}}}, nil
				}
			}
			capture, failure := captureFileRequest(t, openaicompat.New(opts).Model("explicit-endpoint-model"), []message.Message{message.NewToolMessage("call", "read", message.ContentOutput{Value: []message.ToolContentItem{{Type: "file-data", MediaType: "image/png", Data: "c3BlY2ltZW4="}}})})
			if test.want {
				if capture.requests != 1 {
					t.Fatalf("declared transport rejected: %v", failure)
				}
			} else if failure == nil || capture.requests != 0 {
				t.Fatal("undeclared transport inferred support")
			}
		})
	}
}

func TestExplicitTextFileRepresentations(t *testing.T) {
	for _, adapter := range fileAdapters() {
		if adapter.name != "anthropic" && adapter.name != "compatible" && adapter.name != "cohere-native" && adapter.name != "deepseek" && adapter.name != "openai-responses" {
			continue
		}
		t.Run(adapter.name, func(t *testing.T) {
			for _, test := range []struct {
				name  string
				data  message.FileData
				valid bool
			}{
				{"inline-text", message.FileDataText{Text: "explicit-text-specimen"}, true},
				{"encoded-text", message.FileDataBytes{Data: "ZXhwbGljaXQtdGV4dC1zcGVjaW1lbg=="}, true},
				{"invalid-base64", message.FileDataBytes{Data: "PRIVATE-not-base64"}, false},
				{"invalid-utf8", message.FileDataBytes{Data: "//4="}, false},
			} {
				t.Run(test.name, func(t *testing.T) {
					want := test.valid
					if adapter.name == "openai-responses" {
						want = test.name != "inline-text"
					}
					capture, failure := captureFileRequest(t, adapter.make(), []message.Message{message.NewUserMessageWithParts(message.FilePart{MimeType: "text/plain", Data: test.data})})
					if !want {
						if failure == nil || capture.requests != 0 || strings.Contains(failure.Error(), "PRIVATE") {
							t.Fatal("invalid text source not rejected statically")
						}
						return
					}
					if capture.requests != 1 {
						t.Fatalf("explicit text source rejected: %v", failure)
					}
					if adapter.name != "openai-responses" && (!strings.Contains(string(capture.body), "explicit-text-specimen") || strings.Contains(string(capture.body), "ZXhwbGljaXQtdGV4dC1zcGVjaW1lbg==")) {
						t.Fatal("UTF-8 text file was dropped or left as base64 text")
					}
				})
			}
		})
	}
}
