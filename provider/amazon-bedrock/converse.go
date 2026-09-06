package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/stream"
)

func escapeModelID(id string) string { return strings.ReplaceAll(url.QueryEscape(id), "+", "%20") }

func bedrockModelName(id string) string {
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	for _, prefix := range []string{"us.", "eu.", "apac.", "global."} {
		id = strings.TrimPrefix(id, prefix)
	}
	return id
}

func (m *BedrockLanguageModel) buildConverseRequest(options *stream.CallOptions) ([]byte, []stream.Warning, error) {
	opts, err := provider.ParseProviderOptions[ChatOptions](options.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}
	messages := options.Messages
	if options.ResponseFormat != nil && options.ResponseFormat.Type == "json" {
		messages = provider.InjectJSONInstruction(messages, options.ResponseFormat.Schema)
	}
	var system, turns []map[string]any
	for _, msg := range messages {
		if msg.Role == message.RoleSystem {
			system = append(system, map[string]any{"text": msg.Content.Text})
			continue
		}
		var content []map[string]any
		if msg.Content.Text != "" {
			content = append(content, map[string]any{"text": msg.Content.Text})
		}
		for _, part := range msg.Content.Parts {
			switch p := part.(type) {
			case message.TextPart:
				content = append(content, map[string]any{"text": p.Text})
			case message.ToolCallPart:
				var input map[string]any
				if err := json.Unmarshal(p.Input, &input); err != nil || input == nil {
					return nil, nil, errors.New("Bedrock tool input must be a JSON object")
				}
				content = append(content, map[string]any{"toolUse": map[string]any{"toolUseId": p.ID, "name": p.Name, "input": input}})
			case message.ToolResultPart:
				status := "success"
				switch p.Output.(type) {
				case message.ErrorTextOutput, message.ErrorJSONOutput, message.ExecutionDeniedOutput:
					status = "error"
				}
				content = append(content, map[string]any{"toolResult": map[string]any{"toolUseId": p.ToolCallID, "status": status, "content": []any{map[string]any{"text": message.ToolOutputText(p.Output)}}}})
			default:
				return nil, nil, fmt.Errorf("unsupported Bedrock Converse content %T", part)
			}
		}
		role := string(msg.Role)
		if msg.Role == message.RoleTool {
			role = "user"
		}
		if len(turns) > 0 && turns[len(turns)-1]["role"] == role {
			last := turns[len(turns)-1]
			last["content"] = append(last["content"].([]map[string]any), content...)
		} else {
			turns = append(turns, map[string]any{"role": role, "content": content})
		}
	}
	body := map[string]any{"messages": turns}
	if len(system) > 0 {
		body["system"] = system
	}
	config := map[string]any{}
	if options.MaxOutputTokens != nil {
		config["maxTokens"] = *options.MaxOutputTokens
	}
	if options.Temperature != nil {
		config["temperature"] = *options.Temperature
	}
	if options.TopP != nil {
		config["topP"] = *options.TopP
	}
	if len(options.StopSequences) > 0 {
		config["stopSequences"] = options.StopSequences
	}
	if len(config) > 0 {
		body["inferenceConfig"] = config
	}
	if len(opts.AdditionalModelRequestFields) > 0 {
		body["additionalModelRequestFields"] = opts.AdditionalModelRequestFields
	}
	if len(options.Tools) > 0 {
		var tools []any
		for _, t := range options.Tools {
			if t.IsProviderTool() {
				return nil, nil, errors.New("Bedrock Converse does not support provider-defined tools")
			}
			tools = append(tools, map[string]any{"toolSpec": map[string]any{"name": t.Name, "description": t.Description, "inputSchema": map[string]any{"json": t.InputSchema}}})
		}
		toolConfig := map[string]any{"tools": tools}
		body["toolConfig"] = toolConfig
		if options.ToolChoice != nil {
			kind, name := "", ""
			if s, ok := options.ToolChoice.(string); ok {
				kind = s
				if s != "auto" && s != "required" && s != "none" && s != "" {
					kind = "tool"
					name = s
				}
			} else {
				raw, err := json.Marshal(options.ToolChoice)
				if err != nil {
					return nil, nil, err
				}
				var choice struct {
					Type     string `json:"type"`
					Name     string `json:"name"`
					ToolName string `json:"toolName"`
				}
				if err := json.Unmarshal(raw, &choice); err != nil {
					return nil, nil, err
				}
				kind = choice.Type
				name = choice.Name
				if name == "" {
					name = choice.ToolName
				}
			}
			switch kind {
			case "none":
				delete(body, "toolConfig")
			case "", "auto":
				toolConfig["toolChoice"] = map[string]any{"auto": map[string]any{}}
			case "required", "any":
				toolConfig["toolChoice"] = map[string]any{"any": map[string]any{}}
			case "tool":
				if name == "" {
					return nil, nil, errors.New("Bedrock tool choice requires a name")
				}
				toolConfig["toolChoice"] = map[string]any{"tool": map[string]any{"name": name}}
			default:
				return nil, nil, fmt.Errorf("unsupported Bedrock tool choice %q", kind)
			}
		}
	}
	raw, err := json.Marshal(body)
	return raw, nil, err
}

func (m *BedrockLanguageModel) processConverseStream(ctx context.Context, body io.Reader, events chan<- stream.Event, raw bool) {
	fail := func(err error) { events <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}} }
	events <- stream.Event{Type: stream.EventStartStep, Data: stream.StartStepEvent{}}
	var started, stopped, textStarted, metadata bool
	var reason stream.FinishReason
	var usage stream.Usage
	type call struct{ id, name, input string }
	calls := map[int]*call{}
	for {
		if ctx.Err() != nil {
			fail(ctx.Err())
			return
		}
		h, p, err := readEventFrame(body)
		if err == io.EOF {
			break
		}
		if err != nil {
			fail(err)
			return
		}
		if raw {
			events <- stream.Event{Type: stream.EventRawChunk, Data: stream.RawChunkEvent{RawValue: json.RawMessage(p)}}
		}
		var e struct {
			ContentBlockIndex int `json:"contentBlockIndex"`
			Start             struct {
				ToolUse struct {
					ToolUseID string `json:"toolUseId"`
					Name      string `json:"name"`
				} `json:"toolUse"`
			} `json:"start"`
			Delta struct {
				Text    string `json:"text"`
				ToolUse struct {
					Input string `json:"input"`
				} `json:"toolUse"`
			} `json:"delta"`
			StopReason string `json:"stopReason"`
			Usage      struct {
				InputTokens  *int `json:"inputTokens"`
				OutputTokens *int `json:"outputTokens"`
				CacheRead    *int `json:"cacheReadInputTokens"`
				CacheWrite   *int `json:"cacheWriteInputTokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(p, &e); err != nil {
			fail(err)
			return
		}
		switch h[":event-type"] {
		case "messageStart":
			if started {
				fail(errors.New("overlapping Bedrock messages"))
				return
			}
			started = true
		case "contentBlockStart":
			if !started || stopped || calls[e.ContentBlockIndex] != nil {
				fail(errors.New("Bedrock content after messageStop"))
				return
			}
			if e.Start.ToolUse.ToolUseID != "" {
				c := &call{id: e.Start.ToolUse.ToolUseID, name: e.Start.ToolUse.Name}
				calls[e.ContentBlockIndex] = c
				events <- stream.Event{Type: stream.EventToolInputStart, Data: stream.ToolInputStartEvent{ID: c.id, ToolName: c.name}}
			}
		case "contentBlockDelta":
			if !started || stopped {
				fail(errors.New("Bedrock delta outside message"))
				return
			}
			if e.Delta.Text != "" {
				if !textStarted {
					textStarted = true
					events <- stream.Event{Type: stream.EventTextStart, Data: stream.TextStartEvent{}}
				}
				events <- stream.Event{Type: stream.EventTextDelta, Data: stream.TextDeltaEvent{Text: e.Delta.Text}}
			}
			if e.Delta.ToolUse.Input != "" {
				c := calls[e.ContentBlockIndex]
				if c == nil {
					fail(errors.New("Bedrock tool delta without start"))
					return
				}
				c.input += e.Delta.ToolUse.Input
				events <- stream.Event{Type: stream.EventToolInputDelta, Data: stream.ToolInputDeltaEvent{ID: c.id, Delta: e.Delta.ToolUse.Input}}
			}
		case "contentBlockStop":
			if c := calls[e.ContentBlockIndex]; c != nil {
				if c.input == "" {
					c.input = "{}"
				}
				var object map[string]any
				if err := json.Unmarshal([]byte(c.input), &object); err != nil || object == nil {
					fail(errors.New("invalid Bedrock tool input"))
					return
				}
				events <- stream.Event{Type: stream.EventToolInputEnd, Data: stream.ToolInputEndEvent{ID: c.id}}
				events <- stream.Event{Type: stream.EventToolCall, Data: stream.ToolCallEvent{ToolCallID: c.id, ToolName: c.name, Input: json.RawMessage(c.input)}}
				delete(calls, e.ContentBlockIndex)
			}
		case "messageStop":
			if !started || stopped {
				fail(errors.New("invalid Bedrock messageStop"))
				return
			}
			stopped = true
			switch e.StopReason {
			case "end_turn", "stop_sequence":
				reason = stream.FinishReasonStop
			case "max_tokens":
				reason = stream.FinishReasonLength
			case "tool_use":
				reason = stream.FinishReasonToolCalls
			default:
				reason = stream.FinishReasonOther
			}
		case "metadata":
			if !stopped || metadata || e.Usage.InputTokens == nil || e.Usage.OutputTokens == nil {
				fail(errors.New("invalid Bedrock stream metadata"))
				return
			}
			metadata = true
			usage.InputTokens.Total = e.Usage.InputTokens
			usage.OutputTokens.Total = e.Usage.OutputTokens
			usage.InputTokens.CacheRead = e.Usage.CacheRead
			usage.InputTokens.CacheWrite = e.Usage.CacheWrite
		default:
			fail(fmt.Errorf("unsupported Bedrock stream event %q", h[":event-type"]))
			return
		}
	}
	if !stopped || !metadata || len(calls) > 0 {
		fail(errors.New("incomplete Bedrock Converse stream"))
		return
	}
	if textStarted {
		events <- stream.Event{Type: stream.EventTextEnd, Data: stream.TextEndEvent{}}
	}
	events <- stream.Event{Type: stream.EventFinishStep, Data: stream.FinishStepEvent{FinishReason: reason, Usage: usage}}
	events <- stream.Event{Type: stream.EventFinish, Data: stream.FinishEvent{FinishReason: reason, Usage: usage}}
}
