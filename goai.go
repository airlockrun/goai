// Package goai provides a Go implementation of AI SDK functionality.
// It mirrors the Vercel AI SDK (ai package) for streaming LLM interactions.
package goai

import (
	"context"
	"encoding/json"
	"strings"
	"sync"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// defaultMaxSteps is the default maximum number of tool-calling steps.
const defaultMaxSteps = 1

// StreamText streams text generation from a language model.
// This is the main entry point, equivalent to ai-sdk's streamText().
// When MaxSteps > 1, it automatically executes tools and continues the loop,
// streaming all events from all steps.
func StreamText(ctx context.Context, input stream.Input) (*stream.Result, error) {
	if input.AbortSignal == nil {
		input.AbortSignal = ctx
	}

	// Set default MaxSteps
	maxSteps := input.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	// Build stop conditions
	stopConditions := []StopCondition{StepCountIs(maxSteps)}

	// Create result with collection channels
	var (
		textBuilder       strings.Builder
		allToolCalls      []stream.ToolCall
		allToolResults    []stream.ToolResultEvent
		allSteps          []StepResult
		finalFinishReason stream.FinishReason
		finalStepText     string
		totalUsage        stream.Usage
		parsedOutput      any
		mu                sync.Mutex
		done              = make(chan struct{})
	)

	// Fan-out: create a channel for external consumption
	fullStream := make(chan stream.Event, 100)

	// Start the multi-step streaming loop in a goroutine
	go func() {
		defer close(fullStream)
		defer close(done)

		allMessages := make([]message.Message, len(input.Messages))
		copy(allMessages, input.Messages)

		currentInput := input
		currentInput.Messages = allMessages

		for {
			// Build CallOptions from Input (this is what providers receive)
			callOptions := buildCallOptions(&currentInput)

			// Get events channel from model
			events, err := currentInput.Model.Stream(ctx, callOptions)
			if err != nil {
				fullStream <- stream.Event{
					Type: stream.EventError,
					Data: stream.ErrorEvent{Error: err},
				}
				return
			}

			// Emit start step event
			fullStream <- stream.Event{
				Type: stream.EventStartStep,
				Data: stream.StartStepEvent{},
			}

			// Collect results for this step
			var (
				stepTextBuilder  strings.Builder
				stepReasoning    []ReasoningContentPart
				stepToolCalls    []stream.ToolCall
				stepFinishReason stream.FinishReason
				stepUsage        stream.Usage
			)

			for event := range events {
				// Forward event to consumer
				fullStream <- event

				// Collect for this step
				switch e := event.Data.(type) {
				case stream.TextDeltaEvent:
					stepTextBuilder.WriteString(e.Text)
				case stream.ReasoningDeltaEvent:
					if len(stepReasoning) == 0 || stepReasoning[len(stepReasoning)-1].ID != e.ID {
						stepReasoning = append(stepReasoning, ReasoningContentPart{ID: e.ID, Text: e.Text})
					} else {
						stepReasoning[len(stepReasoning)-1].Text += e.Text
					}
				case stream.ReasoningEndEvent:
					// Capture provider metadata (contains encrypted_content)
					if len(stepReasoning) > 0 && stepReasoning[len(stepReasoning)-1].ID == e.ID {
						stepReasoning[len(stepReasoning)-1].ProviderOptions = e.ProviderMetadata
					} else if e.ProviderMetadata != nil {
						// Create reasoning part if we only got the end event
						stepReasoning = append(stepReasoning, ReasoningContentPart{
							ID:              e.ID,
							ProviderOptions: e.ProviderMetadata,
						})
					}
				case stream.ToolCallEvent:
					stepToolCalls = append(stepToolCalls, stream.ToolCall{
						ID:    e.ToolCallID,
						Name:  e.ToolName,
						Input: e.Input,
					})
				case stream.FinishEvent:
					stepFinishReason = e.FinishReason
					stepUsage = e.Usage
				case stream.FinishStepEvent:
					if stepFinishReason == "" {
						stepFinishReason = e.FinishReason
					}
					stepUsage = e.Usage
				}
			}

			// Build step content
			stepText := stepTextBuilder.String()
			content := make([]ContentPart, 0)
			if stepText != "" {
				content = append(content, TextContentPart{Text: stepText})
			}
			for _, r := range stepReasoning {
				content = append(content, r)
			}
			for _, tc := range stepToolCalls {
				content = append(content, ToolCallContentPart{ToolCall: tc})
			}

			// Execute tools if finish reason is tool-calls
			var stepToolResults []stream.ToolResultEvent
			if stepFinishReason == stream.FinishReasonToolCalls && len(stepToolCalls) > 0 {
				// Use provided executor or create default LocalExecutor
				executor := input.Executor
				if executor == nil {
					executor = tool.NewLocalExecutor(input.Tools, input.ActiveTools)
				}
				var toolExecErr error
				stepToolResults, toolExecErr = executeTools(ctx, executor, stepToolCalls)
				if toolExecErr != nil {
					// Emit partial results first (tools completed before the error)
					for _, tr := range stepToolResults {
						fullStream <- stream.Event{Type: stream.EventToolResult, Data: tr}
					}
					fullStream <- stream.Event{
						Type: stream.EventError,
						Data: stream.ErrorEvent{Error: toolExecErr},
					}
					return
				}

				// Emit tool result events
				for _, tr := range stepToolResults {
					fullStream <- stream.Event{
						Type: stream.EventToolResult,
						Data: tr,
					}
				}

				// Add tool results to content
				for _, tr := range stepToolResults {
					content = append(content, ToolResultContentPart{ToolResultEvent: tr})
				}
			}

			// Build step result
			stepResult := StepResult{
				Content:      content,
				FinishReason: stepFinishReason,
				Usage:        stepUsage,
			}

			// Build messages for this step
			stepMessages := buildStepMessages(stepText, stepReasoning, stepToolCalls, stepToolResults)
			stepResult.Response.Messages = stepMessages

			// Update accumulated state
			mu.Lock()
			textBuilder.WriteString(stepText)
			allToolCalls = append(allToolCalls, stepToolCalls...)
			allToolResults = append(allToolResults, stepToolResults...)
			allSteps = append(allSteps, stepResult)
			totalUsage.Add(stepUsage)
			finalFinishReason = stepFinishReason
			finalStepText = stepText
			mu.Unlock()

			// Append step messages to all messages
			allMessages = append(allMessages, stepMessages...)

			// Call OnStepFinish callback
			if input.OnStepFinish != nil {
				input.OnStepFinish(&stepResult)
			}

			// Check stop conditions
			mu.Lock()
			shouldStop := IsStopConditionMet(stopConditions, allSteps)
			mu.Unlock()

			if shouldStop {
				break
			}

			// If finish reason is not tool-calls, we're done
			if stepFinishReason != stream.FinishReasonToolCalls {
				break
			}

			// If no tool calls or no tools, we're done
			if len(stepToolCalls) == 0 || len(input.Tools) == 0 {
				break
			}

			// Update messages for next iteration
			currentInput.Messages = allMessages
		}

		// Parse output only if the last step was finished with "stop"
		// (matches ai-sdk generate-text.ts:904-916).
		if input.Output != nil && finalFinishReason == stream.FinishReasonStop {
			out, parseErr := input.Output.ParseComplete(finalStepText, stream.OutputParseContext{
				FinishReason: finalFinishReason,
				Usage:        totalUsage,
			})
			if parseErr != nil {
				fullStream <- stream.Event{
					Type: stream.EventError,
					Data: stream.ErrorEvent{Error: parseErr},
				}
				return
			}
			mu.Lock()
			parsedOutput = out
			mu.Unlock()
		}

		// Emit final finish event with total usage
		fullStream <- stream.Event{
			Type: stream.EventFinish,
			Data: stream.FinishEvent{
				FinishReason: finalFinishReason,
				Usage:        totalUsage,
			},
		}

		// Call OnFinish callback
		if input.OnFinish != nil {
			mu.Lock()
			stepsData := make([]stream.StepResultData, len(allSteps))
			for i := range allSteps {
				stepsData[i] = &allSteps[i]
			}
			finalStep := &allSteps[len(allSteps)-1]
			mu.Unlock()

			input.OnFinish(stream.OnFinishData{
				Steps:      stepsData,
				TotalUsage: totalUsage,
				FinalStep:  finalStep,
			})
		}
	}()

	result := &stream.Result{
		FullStream: fullStream,
		Text: func() string {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return textBuilder.String()
		},
		ToolCalls: func() []stream.ToolCall {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return allToolCalls
		},
		ToolResults: func() []stream.ToolResultEvent {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return allToolResults
		},
		FinishReason: func() stream.FinishReason {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return finalFinishReason
		},
		Usage: func() stream.Usage {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return totalUsage
		},
		Output: func() any {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return parsedOutput
		},
	}

	return result, nil
}

// GenerateTextResult contains the result of a GenerateText call.
// Equivalent to ai-sdk's GenerateTextResult.
type GenerateTextResult struct {
	// Text is the generated text from the final step.
	Text string

	// ToolCalls contains tool calls from the final step.
	ToolCalls []stream.ToolCall

	// ToolResults contains tool results from the final step.
	ToolResults []stream.ToolResultEvent

	// FinishReason indicates why the final step stopped.
	FinishReason stream.FinishReason

	// Usage contains total token usage across all steps.
	Usage stream.Usage

	// Steps contains all step results when MaxSteps > 1.
	Steps []StepResult

	// Output is the parsed result from Input.Output's ParseComplete.
	// Only populated when Input.Output was set and the final step finished
	// with FinishReasonStop. Type-assert to the expected type.
	Output any

	// Response contains additional response metadata.
	Response GenerateTextResponseMeta
}

// GenerateTextResponseMeta contains response metadata.
type GenerateTextResponseMeta struct {
	// ID is the response ID from the provider.
	ID string

	// Model is the model that was used.
	Model string

	// Messages contains the conversation messages including the response.
	Messages []message.Message
}

// GenerateText generates text from a language model (non-streaming).
// This is equivalent to ai-sdk's generateText().
// It waits for the complete response before returning.
// When MaxSteps > 1, it automatically executes tools and continues the loop.
func GenerateText(ctx context.Context, input stream.Input) (*GenerateTextResult, error) {
	if input.AbortSignal == nil {
		input.AbortSignal = ctx
	}

	// Set default MaxSteps
	maxSteps := input.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	// Build stop conditions (default: stepCountIs(maxSteps))
	stopConditions := []StopCondition{StepCountIs(maxSteps)}

	// Track all steps and accumulated usage
	var (
		allSteps    []StepResult
		totalUsage  stream.Usage
		allMessages = make([]message.Message, len(input.Messages))
	)
	copy(allMessages, input.Messages)

	// Create a working copy of the input for the loop
	currentInput := input
	currentInput.Messages = allMessages

	for {
		// Build CallOptions from Input (this is what providers receive)
		callOptions := buildCallOptions(&currentInput)

		// Get events channel from model
		events, err := currentInput.Model.Stream(ctx, callOptions)
		if err != nil {
			return nil, err
		}

		// Collect results for this step
		var (
			textBuilder   strings.Builder
			reasoning     []ReasoningContentPart
			stepToolCalls []stream.ToolCall
			finishReason  stream.FinishReason
			stepUsage     stream.Usage
			lastError     error
		)

		for event := range events {
			switch e := event.Data.(type) {
			case stream.TextDeltaEvent:
				textBuilder.WriteString(e.Text)
			case stream.ReasoningDeltaEvent:
				// Accumulate reasoning
				if len(reasoning) == 0 || reasoning[len(reasoning)-1].ID != e.ID {
					reasoning = append(reasoning, ReasoningContentPart{ID: e.ID, Text: e.Text})
				} else {
					reasoning[len(reasoning)-1].Text += e.Text
				}
			case stream.ReasoningEndEvent:
				// Capture provider metadata (contains encrypted_content)
				if len(reasoning) > 0 && reasoning[len(reasoning)-1].ID == e.ID {
					reasoning[len(reasoning)-1].ProviderOptions = e.ProviderMetadata
				} else if e.ProviderMetadata != nil {
					// Create reasoning part if we only got the end event
					reasoning = append(reasoning, ReasoningContentPart{
						ID:              e.ID,
						ProviderOptions: e.ProviderMetadata,
					})
				}
			case stream.ToolCallEvent:
				stepToolCalls = append(stepToolCalls, stream.ToolCall{
					ID:    e.ToolCallID,
					Name:  e.ToolName,
					Input: e.Input,
				})
			case stream.FinishEvent:
				finishReason = e.FinishReason
				stepUsage = e.Usage
			case stream.FinishStepEvent:
				if finishReason == "" {
					finishReason = e.FinishReason
				}
				stepUsage = e.Usage
			case stream.ErrorEvent:
				lastError = e.Error
			}
		}

		if lastError != nil {
			return nil, lastError
		}

		// Build step content
		text := textBuilder.String()
		content := make([]ContentPart, 0)

		// Add text content
		if text != "" {
			content = append(content, TextContentPart{Text: text})
		}

		// Add reasoning content
		for _, r := range reasoning {
			content = append(content, r)
		}

		// Add tool calls to content
		for _, tc := range stepToolCalls {
			content = append(content, ToolCallContentPart{ToolCall: tc})
		}

		// Execute tools if finish reason is tool-calls
		var stepToolResults []stream.ToolResultEvent
		if finishReason == stream.FinishReasonToolCalls && len(stepToolCalls) > 0 {
			// Use provided executor or create default LocalExecutor
			executor := input.Executor
			if executor == nil {
				executor = tool.NewLocalExecutor(input.Tools, input.ActiveTools)
			}
			var toolExecErr error
			stepToolResults, toolExecErr = executeTools(ctx, executor, stepToolCalls)
			if toolExecErr != nil {
				return nil, toolExecErr
			}

			// Add tool results to content
			for _, tr := range stepToolResults {
				content = append(content, ToolResultContentPart{ToolResultEvent: tr})
			}
		}

		// Build step result
		stepResult := StepResult{
			Content:      content,
			FinishReason: finishReason,
			Usage:        stepUsage,
		}

		// Accumulate usage
		totalUsage.Add(stepUsage)

		// Build messages for this step
		stepMessages := buildStepMessages(text, reasoning, stepToolCalls, stepToolResults)
		stepResult.Response.Messages = stepMessages

		// Append step messages to all messages
		allMessages = append(allMessages, stepMessages...)

		// Add step to list
		allSteps = append(allSteps, stepResult)

		// Call OnStepFinish callback
		if input.OnStepFinish != nil {
			input.OnStepFinish(&stepResult)
		}

		// Check stop conditions
		if IsStopConditionMet(stopConditions, allSteps) {
			break
		}

		// If finish reason is not tool-calls, we're done
		if finishReason != stream.FinishReasonToolCalls {
			break
		}

		// If no tool calls or no tools, we're done
		if len(stepToolCalls) == 0 || len(input.Tools) == 0 {
			break
		}

		// Update messages for next iteration
		currentInput.Messages = allMessages
	}

	// Get final step
	finalStep := allSteps[len(allSteps)-1]

	// Parse output only if the last step was finished with "stop"
	// (matches ai-sdk generate-text.ts:904-916).
	var parsedOutput any
	if input.Output != nil && finalStep.FinishReason == stream.FinishReasonStop {
		var parseErr error
		parsedOutput, parseErr = input.Output.ParseComplete(finalStep.Text(), stream.OutputParseContext{
			FinishReason: finalStep.FinishReason,
			Usage:        totalUsage,
		})
		if parseErr != nil {
			return nil, parseErr
		}
	}

	// Build final result
	result := &GenerateTextResult{
		Text:         finalStep.Text(),
		ToolCalls:    finalStep.ToolCalls(),
		ToolResults:  finalStep.ToolResults(),
		FinishReason: finalStep.FinishReason,
		Usage:        totalUsage,
		Steps:        allSteps,
		Output:       parsedOutput,
		Response: GenerateTextResponseMeta{
			Model:    input.Model.ID(),
			Messages: allMessages,
		},
	}

	// Call OnFinish callback
	if input.OnFinish != nil {
		stepsData := make([]stream.StepResultData, len(allSteps))
		for i := range allSteps {
			stepsData[i] = &allSteps[i]
		}
		input.OnFinish(stream.OnFinishData{
			Steps:      stepsData,
			TotalUsage: totalUsage,
			FinalStep:  &finalStep,
		})
	}

	return result, nil
}

// buildCallOptions converts a stream.Input to stream.CallOptions.
// This is called at the core level to prepare the provider-facing input.
// It converts the Tools map to an ordered slice and copies all other options.
// When Output is set, its ResponseFormat is sent on every step (matching
// ai-sdk's generate-text.ts:578).
func buildCallOptions(input *stream.Input) *stream.CallOptions {
	opts := &stream.CallOptions{
		Messages:         input.Messages,
		Tools:            input.Tools.Ordered(input.ActiveTools),
		ToolChoice:       input.ToolChoice,
		Temperature:      input.Temperature,
		TopP:             input.TopP,
		TopK:             input.TopK,
		MaxOutputTokens:  input.MaxOutputTokens,
		StopSequences:    input.StopSequences,
		AbortSignal:      input.AbortSignal,
		Headers:          input.Headers,
		ProviderOptions:  input.ProviderOptions,
		IncludeRawChunks: input.IncludeRawChunks,
		Reasoning:        input.Reasoning,
	}
	if input.Output != nil {
		opts.ResponseFormat = input.Output.ResponseFormat()
	}
	return opts
}

// executeTools executes all tool calls using the provided executor and returns the results.
func executeTools(ctx context.Context, executor tool.Executor, toolCalls []stream.ToolCall) ([]stream.ToolResultEvent, error) {
	results := make([]stream.ToolResultEvent, 0, len(toolCalls))

	for _, tc := range toolCalls {
		// Execute via the executor
		resp, err := executor.Execute(ctx, tool.Request{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Input:      tc.Input,
		})

		if err != nil {
			// Fatal errors and context errors propagate up to stop the run
			if ctx.Err() != nil {
				return results, ctx.Err()
			}
			if _, ok := err.(tool.FatalToolError); ok {
				return results, err
			}

			// Normal executor errors: wrap as tool result for model feedback
			results = append(results, stream.ToolResultEvent{
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
				Input:      tc.Input,
				Output: stream.ToolOutput{
					Output: "Error: " + err.Error(),
				},
			})
			continue
		}

		// Skip if tool has no execute function (matches ai-sdk behavior where
		// tools without execute return undefined, which is filtered out)
		if resp.NoExecute {
			continue
		}

		results = append(results, stream.ToolResultEvent{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Input:      tc.Input,
			Output: stream.ToolOutput{
				Output:      resp.Output,
				Title:       resp.Title,
				Metadata:    resp.Metadata,
				Attachments: convertAttachments(resp.Attachments),
			},
		})
	}

	return results, nil
}

// convertAttachments converts tool attachments to stream attachments.
func convertAttachments(attachments []tool.Attachment) []stream.Attachment {
	if len(attachments) == 0 {
		return nil
	}
	result := make([]stream.Attachment, len(attachments))
	for i, a := range attachments {
		result[i] = stream.Attachment{
			Data:     a.Data,
			MimeType: a.MimeType,
			Filename: a.Filename,
		}
	}
	return result
}

// buildStepMessages builds the messages for a step result.
func buildStepMessages(text string, reasoning []ReasoningContentPart, toolCalls []stream.ToolCall, toolResults []stream.ToolResultEvent) []message.Message {
	var msgs []message.Message

	// Add assistant response
	hasParts := len(toolCalls) > 0 || len(reasoning) > 0
	if hasParts {
		parts := make([]message.Part, 0, len(toolCalls)+len(reasoning)+1)
		if text != "" {
			parts = append(parts, message.TextPart{Text: text})
		}
		// Add reasoning parts
		for _, r := range reasoning {
			parts = append(parts, message.ReasoningPart{
				Text:            r.Text,
				ProviderOptions: r.ProviderOptions,
			})
		}
		// Add tool calls
		for _, tc := range toolCalls {
			input := tc.Input
			// Mirror ai-sdk #14281: if the model emitted invalid JSON
			// for a tool-call input, substitute an empty object so the
			// response message can still be re-serialized on follow-up
			// steps instead of triggering a parse error downstream.
			if len(input) > 0 {
				var probe any
				if err := json.Unmarshal(input, &probe); err != nil {
					input = json.RawMessage("{}")
				} else if _, isObject := probe.(map[string]any); !isObject {
					input = json.RawMessage("{}")
				}
			} else {
				input = json.RawMessage("{}")
			}
			parts = append(parts, message.ToolCallPart{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: input,
			})
		}
		msgs = append(msgs, message.NewAssistantMessageWithParts(parts...))
	} else if text != "" {
		msgs = append(msgs, message.NewAssistantMessage(text))
	}

	// Add tool results as tool messages
	for _, tr := range toolResults {
		toolMsg := message.NewToolMessage(
			tr.ToolCallID,
			tr.ToolName,
			tr.Output.Output,
			false,
		)
		// Convert tool result attachments to message content parts.
		for _, att := range tr.Output.Attachments {
			if strings.HasPrefix(att.MimeType, "image/") {
				toolMsg.Content.Parts = append(toolMsg.Content.Parts,
					message.ImagePart{Image: att.Data, MimeType: att.MimeType})
			} else {
				toolMsg.Content.Parts = append(toolMsg.Content.Parts,
					message.FilePart{Data: att.Data, MimeType: att.MimeType, Filename: att.Filename})
			}
		}
		msgs = append(msgs, toolMsg)
	}

	return msgs
}

// Tool is a convenience function to create a tool definition.
// Equivalent to ai-sdk's tool() function.
func Tool(name, description string, schema json.RawMessage, execute tool.ExecuteFunc) tool.Tool {
	return tool.Tool{
		Name:        name,
		Description: description,
		InputSchema: schema,
		Execute:     execute,
	}
}

// Re-export commonly used types for convenience
type (
	Message       = message.Message
	Part          = message.Part
	TextPart      = message.TextPart
	ToolCallPart  = message.ToolCallPart
	ReasoningPart = message.ReasoningPart
	ToolSet       = tool.Set
	StreamResult  = stream.Result
	StreamEvent   = stream.Event
	EventType     = stream.EventType
)

// Message constructors
var (
	NewSystemMessage             = message.NewSystemMessage
	NewUserMessage               = message.NewUserMessage
	NewAssistantMessage          = message.NewAssistantMessage
	NewAssistantMessageWithParts = message.NewAssistantMessageWithParts
	NewToolMessage               = message.NewToolMessage
)
