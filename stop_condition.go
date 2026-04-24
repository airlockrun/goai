// Package goai provides a Go implementation of AI SDK functionality.
package goai

// StopCondition is a function that determines whether the tool loop should stop.
// It receives the steps taken so far and returns true if the loop should stop.
// Equivalent to ai-sdk's StopCondition type.
type StopCondition func(steps []StepResult) bool

// StepCountIs creates a stop condition that stops after the given number of steps.
// This is the default stop condition with stepCount=1.
// Equivalent to ai-sdk's stepCountIs().
func StepCountIs(stepCount int) StopCondition {
	return func(steps []StepResult) bool {
		return len(steps) >= stepCount
	}
}

// HasToolCall creates a stop condition that stops when the specified tool is called.
// It checks the last step's tool calls for a matching tool name.
// Equivalent to ai-sdk's hasToolCall().
func HasToolCall(toolName string) StopCondition {
	return func(steps []StepResult) bool {
		if len(steps) == 0 {
			return false
		}
		lastStep := steps[len(steps)-1]
		for _, tc := range lastStep.ToolCalls() {
			if tc.Name == toolName {
				return true
			}
		}
		return false
	}
}

// IsStopConditionMet checks if any of the stop conditions are met.
// Returns true if ANY condition returns true (OR logic).
// Equivalent to ai-sdk's isStopConditionMet().
func IsStopConditionMet(stopConditions []StopCondition, steps []StepResult) bool {
	for _, condition := range stopConditions {
		if condition(steps) {
			return true
		}
	}
	return false
}
