// Package goai provides a Go implementation of AI SDK functionality.
// It mirrors the Vercel AI SDK (ai package) for streaming LLM interactions.
package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
	"github.com/airlockrun/goai/tool"
)

// defaultMaxSteps is the default maximum number of tool-calling steps.
const defaultMaxSteps = 1

const skippedAfterDeniedReason = "This tool was not executed because an earlier tool call in the same ordered batch was denied by the user."

// StreamText streams text generation from a language model.
// This is the main entry point, equivalent to ai-sdk's streamText().
// When MaxSteps > 1, it automatically executes tools and continues the loop,
// streaming all events from all steps.
func StreamText(ctx context.Context, input stream.Input) (*stream.Result, error) {
	if input.AbortSignal == nil {
		input.AbortSignal = ctx
	}
	if input.MaxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be >= 0")
	}
	toolCallExecutionMode, err := resolveToolCallExecutionMode(input.ToolCallExecutionMode)
	if err != nil {
		return nil, err
	}
	input.ToolCallExecutionMode = toolCallExecutionMode

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
		allSources        []stream.SourceEvent
		allSteps          []StepResult
		finalFinishReason stream.FinishReason
		finalStepText     string
		totalUsage        stream.Usage
		parsedOutput      any
		terminalErr       error
		mu                sync.Mutex
		done              = make(chan struct{})
	)

	// Fan-out: create a channel for external consumption
	fullStream := make(chan stream.Event, 100)

	// Start the multi-step streaming loop in a goroutine
	go func() {
		defer close(fullStream)
		defer close(done)

		recordError := func(err error) {
			mu.Lock()
			if terminalErr == nil {
				terminalErr = err
				finalFinishReason = stream.FinishReasonError
			}
			mu.Unlock()
			if input.OnError != nil {
				input.OnError(err)
			}
		}

		emitError := func(err error) {
			recordError(err)
			fullStream <- stream.Event{
				Type: stream.EventError,
				Data: stream.ErrorEvent{Error: err},
			}
		}

		allMessages := make([]message.Message, len(input.Messages))
		copy(allMessages, input.Messages)

		currentInput := input
		currentInput.Messages = allMessages

		if input.OnStart != nil {
			input.OnStart(stream.StartData{Messages: allMessages})
		}

		stepNumber := 0
		for {
			// Build CallOptions from Input (this is what providers receive)
			callOptions := buildCallOptions(&currentInput)

			if input.OnStepStart != nil {
				input.OnStepStart(stream.StepStartData{StepNumber: stepNumber, Messages: currentInput.Messages})
			}

			// Retry only errors returned while establishing the provider stream.
			// Start events do not commit output; the first non-start event does.
			events, err := streamWithSetupRetries(ctx, currentInput.Model, callOptions, currentInput.MaxRetries, currentInput.MaxRetriesSet)
			if err != nil {
				emitError(err)
				return
			}

			// Emit start step event
			fullStream <- stream.Event{
				Type: stream.EventStartStep,
				Data: stream.StartStepEvent{},
			}

			// Collect results for this step
			var (
				stepTextBuilder     strings.Builder
				stepReasoning       []ReasoningContentPart
				stepToolCalls       []stream.ToolCall
				providerToolResults []stream.ToolResultEvent
				stepSources         []stream.SourceEvent
				stepFinishReason    stream.FinishReason
				stepUsage           stream.Usage
				replayEvents        []stream.Event
			)

			for event := range events {
				switch event.Data.(type) {
				case stream.TextDeltaEvent, stream.ReasoningDeltaEvent, stream.ReasoningEndEvent, stream.ToolCallEvent, stream.ToolResultEvent, stream.ToolErrorEvent:
					replayEvents = append(replayEvents, event)
				}
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
					refined := e.Input
					if input.RefineToolInput != nil && !e.ProviderExecuted {
						r, refineErr := input.RefineToolInput(e.ToolName, e.Input)
						if refineErr != nil {
							err := fmt.Errorf("refineToolInput(%s): %w", e.ToolName, refineErr)
							mu.Lock()
							stepText := stepTextBuilder.String()
							textBuilder.WriteString(stepText)
							allSources = append(allSources, stepSources...)
							finalStepText = stepText
							mu.Unlock()
							emitError(err)
							return
						}
						refined = r
					}
					stepToolCalls = append(stepToolCalls, stream.ToolCall{
						ProviderExecuted: e.ProviderExecuted,
						ProviderMetadata: e.ProviderMetadata,
						ID:               e.ToolCallID,
						Name:             e.ToolName,
						Input:            refined,
					})
				case stream.ToolResultEvent:
					providerToolResults = append(providerToolResults, e)
				case stream.ToolErrorEvent:
					providerToolResults = append(providerToolResults, stream.ToolResultEvent{ToolCallID: e.ToolCallID, ToolName: e.ToolName, Input: e.Input, Output: e.Output, ProviderExecuted: e.ProviderExecuted, ProviderMetadata: e.ProviderMetadata})
				case stream.SourceEvent:
					stepSources = append(stepSources, e)
				case stream.FinishEvent:
					stepFinishReason = e.FinishReason
					stepUsage = e.Usage.Normalized()
				case stream.FinishStepEvent:
					if stepFinishReason == "" {
						stepFinishReason = e.FinishReason
					}
					stepUsage = e.Usage.Normalized()
				case stream.ErrorEvent:
					mu.Lock()
					stepText := stepTextBuilder.String()
					textBuilder.WriteString(stepText)
					allToolCalls = append(allToolCalls, stepToolCalls...)
					allToolResults = append(allToolResults, providerToolResults...)
					allSources = append(allSources, stepSources...)
					totalUsage.Add(stepUsage)
					finalStepText = stepText
					mu.Unlock()
					recordError(e.Error)
					return
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
			for _, src := range stepSources {
				content = append(content, SourceContentPart{SourceEvent: src})
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
				stepToolResults, toolExecErr = executeTools(ctx, executor, stepToolCalls, input.ToolCallExecutionMode)
				if toolExecErr != nil {
					// Emit partial results first (tools completed before the error)
					for _, tr := range stepToolResults {
						fullStream <- toolOutcomeEvent(tr)
					}
					mu.Lock()
					stepText := stepTextBuilder.String()
					textBuilder.WriteString(stepText)
					allToolCalls = append(allToolCalls, stepToolCalls...)
					allToolResults = append(allToolResults, providerToolResults...)
					allToolResults = append(allToolResults, stepToolResults...)
					allSources = append(allSources, stepSources...)
					totalUsage.Add(stepUsage)
					finalStepText = stepText
					mu.Unlock()
					emitError(toolExecErr)
					return
				}

				// Emit tool result events (kind reflects the outcome variant)
				for _, tr := range stepToolResults {
					fullStream <- toolOutcomeEvent(tr)
				}

				// Add tool results to content
				for _, tr := range stepToolResults {
					content = append(content, ToolResultContentPart{ToolResultEvent: tr})
				}
			}

			stepToolResults = append(providerToolResults, stepToolResults...)
			for _, tr := range providerToolResults {
				content = append(content, ToolResultContentPart{ToolResultEvent: tr})
			}
			// Build step result
			stepResult := StepResult{
				Content:      content,
				FinishReason: stepFinishReason,
				Usage:        stepUsage,
			}

			// Build messages for this step
			stepMessages := buildStepMessages(replayEvents, stepToolCalls, stepToolResults)
			stepResult.Response.Messages = stepMessages

			// Update accumulated state
			mu.Lock()
			textBuilder.WriteString(stepText)
			allToolCalls = append(allToolCalls, stepToolCalls...)
			allToolResults = append(allToolResults, stepToolResults...)
			allSources = append(allSources, stepSources...)
			allSteps = append(allSteps, stepResult)
			totalUsage.Add(stepUsage)
			finalFinishReason = stepFinishReason
			finalStepText = stepText
			mu.Unlock()

			// Append step messages to all messages
			allMessages = append(allMessages, stepMessages...)

			// Call OnStepEnd callback
			if input.OnStepEnd != nil {
				input.OnStepEnd(&stepResult)
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
			if !hasLocalToolCalls(stepToolCalls) || len(input.Tools) == 0 {
				break
			}

			// Update messages for next iteration
			currentInput.Messages = allMessages
			stepNumber++
		}

		// Parse output only if the last step was finished with "stop"
		// (matches ai-sdk generate-text.ts:904-916).
		if input.Output != nil && finalFinishReason == stream.FinishReasonStop {
			out, parseErr := input.Output.ParseComplete(finalStepText, stream.OutputParseContext{
				FinishReason: finalFinishReason,
				Usage:        totalUsage,
			})
			if parseErr != nil {
				emitError(parseErr)
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

		// Call OnEnd callback
		if input.OnEnd != nil {
			mu.Lock()
			stepsData := make([]stream.StepResultData, len(allSteps))
			for i := range allSteps {
				stepsData[i] = &allSteps[i]
			}
			finalStep := &allSteps[len(allSteps)-1]
			mu.Unlock()

			input.OnEnd(stream.OnEndData{
				Steps:      stepsData,
				TotalUsage: totalUsage,
				FinalStep:  finalStep,
			})
		}
	}()

	result := &stream.Result{
		FullStream: fullStream,
		Text: func() (string, error) {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return textBuilder.String(), terminalErr
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
		Sources: func() []stream.SourceEvent {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return allSources
		},
		FinishReason: func() (stream.FinishReason, error) {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return finalFinishReason, terminalErr
		},
		Usage: func() (stream.Usage, error) {
			<-done
			mu.Lock()
			defer mu.Unlock()
			return totalUsage, terminalErr
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

func streamWithSetupRetries(ctx context.Context, model stream.Model, options *stream.CallOptions, configuredMaxRetries int, maxRetriesSet bool) (<-chan stream.Event, error) {
	maxRetries := 2
	if maxRetriesSet || configuredMaxRetries != 0 {
		maxRetries = configuredMaxRetries
	} else if configured, ok := model.(interface{ SetupMaxRetries() (int, bool) }); ok {
		if modelMaxRetries, set := configured.SetupMaxRetries(); set {
			maxRetries = modelMaxRetries
		}
	}
	if maxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be >= 0")
	}

	for attempt := 0; ; attempt++ {
		events, err := model.Stream(ctx, options)
		if err == nil {
			events, err = inspectStreamSetup(ctx, events)
			if err == nil {
				return events, nil
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		var apiErr *goaierrors.APICallError
		if attempt >= maxRetries || !errors.As(err, &apiErr) || !apiErr.IsRetryable {
			return nil, err
		}

		delay := retryDelay(apiErr, attempt)
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// inspectStreamSetup adapts providers that begin the HTTP request in their
// stream goroutine. A retryable API error before any content is still a setup
// failure; once any event other than request/step lifecycle framing is observed,
// the stream is committed and subsequent failures are never replayed.
func inspectStreamSetup(ctx context.Context, events <-chan stream.Event) (<-chan stream.Event, error) {
	if events == nil {
		return nil, errors.New("model returned nil event stream")
	}
	var prefix []stream.Event
	for {
		select {
		case event, ok := <-events:
			if !ok {
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				return nil, fmt.Errorf("%w: model stream closed without content or finish", goaierrors.ErrInvalidResponse)
			}
			prefix = append(prefix, event)
			switch event.Type {
			case stream.EventStart, stream.EventStartStep, stream.EventRawChunk:
				continue
			case stream.EventError:
				if eventErr, ok := event.Data.(stream.ErrorEvent); ok {
					go drainEvents(ctx, events)
					if eventErr.Error == nil {
						return nil, fmt.Errorf("%w: model emitted an empty error", goaierrors.ErrInvalidResponse)
					}
					return nil, eventErr.Error
				}
			}
			return prependEvents(ctx, prefix, events), nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func prependEvents(ctx context.Context, prefix []stream.Event, events <-chan stream.Event) <-chan stream.Event {
	out := make(chan stream.Event, len(prefix))
	go func() {
		defer close(out)
		hasOutput := false
		forward := func(event stream.Event) {
			switch e := event.Data.(type) {
			case stream.TextDeltaEvent:
				hasOutput = hasOutput || e.Text != ""
			case stream.ReasoningDeltaEvent:
				hasOutput = hasOutput || e.Text != ""
			case stream.ReasoningEndEvent:
				// Encrypted reasoning can be carried entirely in provider metadata.
				hasOutput = hasOutput || len(e.ProviderMetadata) > 0
			case stream.ToolCallEvent, stream.ToolResultEvent, stream.ToolErrorEvent, stream.ToolOutputDeniedEvent, stream.SourceEvent, stream.FinishEvent, stream.FinishStepEvent:
				hasOutput = true
			case stream.ErrorEvent:
				hasOutput = true
				if e.Error == nil {
					event.Data = stream.ErrorEvent{Error: fmt.Errorf("%w: model emitted an empty error", goaierrors.ErrInvalidResponse)}
				}
			}
			out <- event
		}
		for _, event := range prefix {
			forward(event)
		}
		if events != nil {
			for event := range events {
				forward(event)
			}
		}
		if !hasOutput {
			err := ctx.Err()
			if err == nil {
				err = fmt.Errorf("%w: model stream closed without content or finish", goaierrors.ErrInvalidResponse)
			}
			out <- stream.Event{Type: stream.EventError, Data: stream.ErrorEvent{Error: err}}
		}
	}()
	return out
}

func drainEvents(ctx context.Context, events <-chan stream.Event) {
	for {
		select {
		case _, ok := <-events:
			if !ok {
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

func retryDelay(apiErr *goaierrors.APICallError, attempt int) time.Duration {
	defaultDelay := time.Duration(1<<attempt) * 2 * time.Second
	for name, value := range apiErr.ResponseHeaders {
		if !strings.EqualFold(name, "Retry-After-Ms") {
			continue
		}
		if milliseconds, err := strconv.ParseFloat(value, 64); err == nil {
			if delay := time.Duration(milliseconds * float64(time.Millisecond)); delay >= 0 && (delay < time.Minute || delay < defaultDelay) {
				return delay
			}
		}
	}
	for name, value := range apiErr.ResponseHeaders {
		if !strings.EqualFold(name, "Retry-After") {
			continue
		}
		if seconds, err := strconv.ParseFloat(value, 64); err == nil {
			if delay := time.Duration(seconds * float64(time.Second)); delay >= 0 && (delay < time.Minute || delay < defaultDelay) {
				return delay
			}
		}
		if at, err := http.ParseTime(value); err == nil {
			if delay := time.Until(at); delay > 0 && delay < time.Minute {
				return delay
			}
		}
	}
	return defaultDelay
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

	// Sources contains citation sources (URLs / documents) from the final
	// step — emitted by hosted search/retrieval tools. Mirrors ai-sdk's
	// GenerateTextResult.sources.
	Sources []stream.SourceEvent

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
	if input.MaxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be >= 0")
	}
	toolCallExecutionMode, err := resolveToolCallExecutionMode(input.ToolCallExecutionMode)
	if err != nil {
		return nil, err
	}
	input.ToolCallExecutionMode = toolCallExecutionMode

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

	if input.OnStart != nil {
		input.OnStart(stream.StartData{Messages: allMessages})
	}

	stepNumber := 0
	for {
		// Build CallOptions from Input (this is what providers receive)
		callOptions := buildCallOptions(&currentInput)

		if input.OnStepStart != nil {
			input.OnStepStart(stream.StepStartData{StepNumber: stepNumber, Messages: currentInput.Messages})
		}

		// Get events channel from model, retrying only stream setup failures.
		events, err := streamWithSetupRetries(ctx, currentInput.Model, callOptions, currentInput.MaxRetries, currentInput.MaxRetriesSet)
		if err != nil {
			return nil, err
		}

		// Collect results for this step
		var (
			textBuilder         strings.Builder
			reasoning           []ReasoningContentPart
			stepToolCalls       []stream.ToolCall
			providerToolResults []stream.ToolResultEvent
			stepSources         []stream.SourceEvent
			finishReason        stream.FinishReason
			stepUsage           stream.Usage
			lastError           error
			replayEvents        []stream.Event
		)

		for event := range events {
			switch event.Data.(type) {
			case stream.TextDeltaEvent, stream.ReasoningDeltaEvent, stream.ReasoningEndEvent, stream.ToolCallEvent, stream.ToolResultEvent, stream.ToolErrorEvent:
				replayEvents = append(replayEvents, event)
			}
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
				refined := e.Input
				if input.RefineToolInput != nil && !e.ProviderExecuted {
					r, err := input.RefineToolInput(e.ToolName, e.Input)
					if err != nil {
						lastError = fmt.Errorf("refineToolInput(%s): %w", e.ToolName, err)
						continue
					}
					refined = r
				}
				stepToolCalls = append(stepToolCalls, stream.ToolCall{
					ProviderExecuted: e.ProviderExecuted,
					ProviderMetadata: e.ProviderMetadata,
					ID:               e.ToolCallID,
					Name:             e.ToolName,
					Input:            refined,
				})
			case stream.ToolResultEvent:
				providerToolResults = append(providerToolResults, e)
			case stream.ToolErrorEvent:
				providerToolResults = append(providerToolResults, stream.ToolResultEvent{ToolCallID: e.ToolCallID, ToolName: e.ToolName, Input: e.Input, Output: e.Output, ProviderExecuted: e.ProviderExecuted, ProviderMetadata: e.ProviderMetadata})
			case stream.SourceEvent:
				stepSources = append(stepSources, e)
			case stream.FinishEvent:
				finishReason = e.FinishReason
				stepUsage = e.Usage.Normalized()
			case stream.FinishStepEvent:
				if finishReason == "" {
					finishReason = e.FinishReason
				}
				stepUsage = e.Usage.Normalized()
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

		// Add cited sources to content (web_search / google_search / file
		// citations). Mirrors ai-sdk's StepResult.sources.
		for _, src := range stepSources {
			content = append(content, SourceContentPart{SourceEvent: src})
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
			stepToolResults, toolExecErr = executeTools(ctx, executor, stepToolCalls, input.ToolCallExecutionMode)
			if toolExecErr != nil {
				return nil, toolExecErr
			}

			// Add tool results to content
			for _, tr := range stepToolResults {
				content = append(content, ToolResultContentPart{ToolResultEvent: tr})
			}
		}

		stepToolResults = append(providerToolResults, stepToolResults...)
		for _, tr := range providerToolResults {
			content = append(content, ToolResultContentPart{ToolResultEvent: tr})
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
		stepMessages := buildStepMessages(replayEvents, stepToolCalls, stepToolResults)
		stepResult.Response.Messages = stepMessages

		// Append step messages to all messages
		allMessages = append(allMessages, stepMessages...)

		// Add step to list
		allSteps = append(allSteps, stepResult)

		// Call OnStepEnd callback
		if input.OnStepEnd != nil {
			input.OnStepEnd(&stepResult)
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
		if !hasLocalToolCalls(stepToolCalls) || len(input.Tools) == 0 {
			break
		}

		// Update messages for next iteration
		currentInput.Messages = allMessages
		stepNumber++
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
		Sources:      finalStep.Sources(),
		FinishReason: finalStep.FinishReason,
		Usage:        totalUsage,
		Steps:        allSteps,
		Output:       parsedOutput,
		Response: GenerateTextResponseMeta{
			Model:    input.Model.ID(),
			Messages: allMessages,
		},
	}

	// Call OnEnd callback
	if input.OnEnd != nil {
		stepsData := make([]stream.StepResultData, len(allSteps))
		for i := range allSteps {
			stepsData[i] = &allSteps[i]
		}
		input.OnEnd(stream.OnEndData{
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
	messages := input.Messages
	// Instructions reaches providers as the leading system message; goai's
	// converters derive their system block from RoleSystem messages.
	if input.Instructions != "" {
		messages = append([]message.Message{message.NewSystemMessage(input.Instructions)}, messages...)
	}
	opts := &stream.CallOptions{
		Messages:         messages,
		Tools:            tool.ApplyToolOrder(input.Tools.Ordered(input.ActiveTools), input.ToolOrder),
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

func resolveToolCallExecutionMode(mode stream.ToolCallExecutionMode) (stream.ToolCallExecutionMode, error) {
	switch mode {
	case "", stream.ToolCallExecutionSync:
		return stream.ToolCallExecutionSync, nil
	case stream.ToolCallExecutionAsync:
		return stream.ToolCallExecutionAsync, nil
	default:
		return "", fmt.Errorf("toolCallExecutionMode must be %q or %q", stream.ToolCallExecutionSync, stream.ToolCallExecutionAsync)
	}
}

func hasLocalToolCalls(calls []stream.ToolCall) bool {
	for _, call := range calls {
		if !call.ProviderExecuted {
			return true
		}
	}
	return false
}

// executeTools executes local tool calls using the provided executor and returns
// results in model call order.
func executeTools(ctx context.Context, executor tool.Executor, toolCalls []stream.ToolCall, mode stream.ToolCallExecutionMode) ([]stream.ToolResultEvent, error) {
	localCalls := make([]stream.ToolCall, 0, len(toolCalls))
	for _, tc := range toolCalls {
		if !tc.ProviderExecuted {
			localCalls = append(localCalls, tc)
		}
	}
	toolCalls = localCalls
	if mode == stream.ToolCallExecutionAsync {
		return executeToolsAsync(ctx, executor, toolCalls)
	}
	return executeToolsSync(ctx, executor, toolCalls)
}

func executeToolsSync(ctx context.Context, executor tool.Executor, toolCalls []stream.ToolCall) ([]stream.ToolResultEvent, error) {
	results := make([]stream.ToolResultEvent, 0, len(toolCalls))

	for i, tc := range toolCalls {
		result, err := executeTool(ctx, executor, tc)
		if err != nil {
			return results, err
		}
		if result == nil {
			continue
		}

		results = append(results, *result)
		if message.ToolOutcome(result.Output) == "denied" {
			for _, skipped := range toolCalls[i+1:] {
				results = append(results, stream.ToolResultEvent{
					ToolCallID: skipped.ID,
					ToolName:   skipped.Name,
					Input:      skipped.Input,
					Output:     message.ExecutionDeniedOutput{Reason: skippedAfterDeniedReason},
				})
			}
			return results, nil
		}
	}

	return results, nil
}

type toolExecutionResult struct {
	result *stream.ToolResultEvent
	err    error
}

func executeToolsAsync(ctx context.Context, executor tool.Executor, toolCalls []stream.ToolCall) ([]stream.ToolResultEvent, error) {
	settled := make([]toolExecutionResult, len(toolCalls))
	var wg sync.WaitGroup
	for i, tc := range toolCalls {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := executeTool(ctx, executor, tc)
			settled[i] = toolExecutionResult{result: result, err: err}
		}()
	}
	wg.Wait()

	results := make([]stream.ToolResultEvent, 0, len(toolCalls))
	var executionErr error
	for _, outcome := range settled {
		if outcome.result != nil {
			results = append(results, *outcome.result)
		}
		if executionErr == nil && outcome.err != nil && !errors.Is(outcome.err, context.Canceled) && !errors.Is(outcome.err, context.DeadlineExceeded) {
			executionErr = outcome.err
		}
	}
	if executionErr != nil {
		return results, executionErr
	}
	if ctx.Err() != nil {
		return results, ctx.Err()
	}
	for _, outcome := range settled {
		if outcome.err != nil {
			return results, outcome.err
		}
	}
	return results, nil
}

func executeTool(ctx context.Context, executor tool.Executor, tc stream.ToolCall) (*stream.ToolResultEvent, error) {
	resp, err := executor.Execute(ctx, tool.Request{
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Input:      tc.Input,
	})
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		var fatal tool.FatalToolError
		if errors.As(err, &fatal) && fatal.FatalToolError() {
			return nil, err
		}
		return &stream.ToolResultEvent{
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Input:      tc.Input,
			Output:     tool.OutputForError(err),
		}, nil
	}
	if resp.NoExecute {
		return nil, nil
	}
	return &stream.ToolResultEvent{
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Input:      tc.Input,
		Output:     outputFromResponse(resp),
		Title:      resp.Title,
		Metadata:   resp.Metadata,
	}, nil
}

// outputFromResponse maps a tool.Response to the discriminated
// ToolResultOutput: denied → execution-denied; is-error → error-text;
// otherwise the success output (text or multipart content).
func outputFromResponse(resp tool.Response) message.ToolResultOutput {
	if resp.Denied {
		return message.ExecutionDeniedOutput{Reason: resp.DeniedReason}
	}
	if resp.IsError {
		v := resp.Error
		if v == "" {
			v = resp.Output
		}
		return message.ErrorTextOutput{Value: v}
	}
	return tool.SuccessOutput(tool.Result{
		Output:      resp.Output,
		Title:       resp.Title,
		Metadata:    resp.Metadata,
		Attachments: resp.Attachments,
	})
}

// toolOutcomeEvent wraps a collected tool result into the live stream event
// whose kind matches its discriminated outcome: success → tool-result,
// error → tool-error, execution-denied → tool-output-denied. Consumers that
// branch on event type (UI, audit) get the right signal; the persisted
// message keeps the full union in ToolResultPart.Output regardless.
func toolOutcomeEvent(tr stream.ToolResultEvent) stream.Event {
	return stream.ToolOutcomeEvent(tr.ToolCallID, tr.ToolName, tr.Input, tr.Output, tr.Title, tr.Metadata)
}

// buildStepMessages builds the messages for a step result.
func buildStepMessages(events []stream.Event, toolCalls []stream.ToolCall, toolResults []stream.ToolResultEvent) []message.Message {
	var msgs []message.Message
	var parts []message.Part
	reasoningIndexes := make(map[string]int)
	callIndex, resultIndex := 0, 0
	var text strings.Builder
	flushText := func() {
		if text.Len() > 0 {
			parts = append(parts, message.TextPart{Text: text.String()})
			text.Reset()
		}
	}
	// Preserve the provider's transcript order, including server-tool rounds.
	for _, event := range events {
		if e, ok := event.Data.(stream.TextDeltaEvent); ok {
			text.WriteString(e.Text)
			continue
		}
		flushText()
		switch e := event.Data.(type) {
		case stream.ReasoningDeltaEvent:
			i, ok := reasoningIndexes[e.ID]
			if !ok {
				i = len(parts)
				reasoningIndexes[e.ID] = i
				parts = append(parts, message.ReasoningPart{})
			}
			p := parts[i].(message.ReasoningPart)
			p.Text += e.Text
			parts[i] = p
		case stream.ReasoningEndEvent:
			i, ok := reasoningIndexes[e.ID]
			if !ok {
				if e.ProviderMetadata == nil {
					continue
				}
				i = len(parts)
				reasoningIndexes[e.ID] = i
				parts = append(parts, message.ReasoningPart{})
			}
			p := parts[i].(message.ReasoningPart)
			p.ProviderOptions = e.ProviderMetadata
			parts[i] = p
		case stream.ToolCallEvent:
			tc := toolCalls[callIndex]
			callIndex++
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
				ProviderExecuted: tc.ProviderExecuted,
				ProviderOptions:  tc.ProviderMetadata,
				ID:               tc.ID,
				Name:             tc.Name,
				Input:            input,
			})
		case stream.ToolResultEvent, stream.ToolErrorEvent:
			tr := toolResults[resultIndex]
			resultIndex++
			if tr.ProviderExecuted {
				parts = append(parts, message.ToolResultPart{ToolCallID: tr.ToolCallID, ToolName: tr.ToolName, Output: tr.Output, ProviderExecuted: true, ProviderOptions: tr.ProviderMetadata})
			}
		}
	}
	flushText()
	if len(parts) == 1 {
		if text, ok := parts[0].(message.TextPart); ok {
			msgs = append(msgs, message.NewAssistantMessage(text.Text))
		} else {
			msgs = append(msgs, message.NewAssistantMessageWithParts(parts...))
		}
	} else if len(parts) > 0 {
		msgs = append(msgs, message.NewAssistantMessageWithParts(parts...))
	}

	// Add tool results as tool messages — the discriminated Output (text /
	// json / content / error-* / execution-denied) is carried as-is, so a
	// content output keeps its file/image items inside the tool-result part
	// rather than being flattened into separate message parts.
	for _, tr := range toolResults {
		if !tr.ProviderExecuted {
			msgs = append(msgs, message.NewToolMessage(tr.ToolCallID, tr.ToolName, tr.Output))
		}
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
	Message               = message.Message
	Part                  = message.Part
	TextPart              = message.TextPart
	ToolCallPart          = message.ToolCallPart
	ToolResultPart        = message.ToolResultPart
	ReasoningPart         = message.ReasoningPart
	ToolResultOutput      = message.ToolResultOutput
	TextOutput            = message.TextOutput
	JSONOutput            = message.JSONOutput
	ErrorTextOutput       = message.ErrorTextOutput
	ErrorJSONOutput       = message.ErrorJSONOutput
	ExecutionDeniedOutput = message.ExecutionDeniedOutput
	ContentOutput         = message.ContentOutput
	ToolContentItem       = message.ToolContentItem
	ToolSet               = tool.Set
	StreamResult          = stream.Result
	StreamEvent           = stream.Event
	EventType             = stream.EventType
)

// Message constructors
var (
	NewSystemMessage             = message.NewSystemMessage
	NewUserMessage               = message.NewUserMessage
	NewAssistantMessage          = message.NewAssistantMessage
	NewAssistantMessageWithParts = message.NewAssistantMessageWithParts
	NewToolMessage               = message.NewToolMessage
	NewToolResultText            = message.NewToolResultText
	NewToolResultJSON            = message.NewToolResultJSON
	NewToolResultDenied          = message.NewToolResultDenied
)

// Tool-output classifiers (heuristic-free, structured).
var (
	ToolOutputText    = message.ToolOutputText
	ToolOutputWire    = message.ToolOutputWire
	ToolOutputIsError = message.ToolOutputIsError
	ToolOutcome       = message.ToolOutcome
)
