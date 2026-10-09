# Mock model

`testutil.NewMockModel(config MockConfig) (*MockModel, error)` creates an
in-process `stream.Model` for offline development and adapter tests.

```go
exact := "ping"
model, err := testutil.NewMockModel(testutil.MockConfig{
    ID: "test-model",
    Default: &testutil.MockResponse{Text: "default answer"},
    Rules: []testutil.MockRule{
        {LastUserText: &exact, Response: testutil.MockResponse{Text: "pong"}},
        {
            Match: func(input *stream.CallOptions) bool {
                return input.ToolChoice == "none"
            },
            Response: testutil.MockResponse{Text: "text-only answer"},
        },
    },
})
if err != nil {
    return err
}
var selected stream.Model = model
// Use selected in stream.Input.Model or Sol's injected model slot.
```

## API and matching

- `MockConfig`: required `ID string`, optional `Default *MockResponse`, ordered
  `Rules []MockRule`, optional fallback sequence `Responses []MockResponse`, and
  optional `Stream func(context.Context, *stream.CallOptions) (<-chan stream.Event, error)`.
- `MockRule`: `LastUserText *string`, `Match func(*stream.CallOptions) bool`,
  and `Response MockResponse`. At least one matcher is required; if both are
  provided, both must match. The first matching rule wins.
- Exact matching finds the last user message even when assistant/tool messages
  follow it. Multipart text parts concatenate without separators; file and
  reasoning parts are ignored. An empty string matches an empty user message,
  but does not match a request without a user message.
- Rules take precedence over the response sequence. Unmatched calls consume the
  next `Responses` entry, repeating the last entry when exhausted. The sequence
  takes precedence over `Default`. Matching rules do not consume sequence entries.
- A nil default and empty sequence return `ErrMockNoMatch` when no rule matches.
  `&MockResponse{}` is an explicit empty-text fallback.
- `MockResponse`: `Text string`, `ToolCalls []stream.ToolCall`, `Usage stream.Usage`,
  and optional `Events []stream.Event`.
  Emits start, text-start, text-delta, text-end, optional tool-call events, and
  finish. Finish reason is `tool-calls` when calls are present, otherwise `stop`.
  Usage is explicit; unset counts remain unreported. Tool calls can drive GoAI's
  real multi-step tool loop; predicates can match subsequent tool-result messages.
- Non-nil `Events` replaces all generated events with the exact script. An empty
  non-nil slice produces an empty stream. Event payloads are copied; error values
  retain their identity. Scripted tool-call inputs can intentionally contain
  malformed JSON to exercise validation and repair paths.

## Runtime configuration

`Configure(config MockConfig) error` atomically replaces the hook, rules, default, and
sequence. Supply the same `ID` used at construction; IDs are immutable. A
successful configuration restarts its sequence and preserves request history.
Invalid configuration leaves behavior and sequence position unchanged. Each
stream captures a configuration before running predicates, so reconfiguration
does not change its response halfway through. `Configure`, `Stream`, `Requests`,
and `ResetRequests` are safe to call concurrently.

```go
err := model.Configure(testutil.MockConfig{
    ID: model.ID(),
    Responses: []testutil.MockResponse{
        {Events: testutil.MockToolCallResponse("call-1", "lookup", map[string]any{}, testutil.MockUsage(1, 1))},
        {Text: "done"},
    },
})
```

## Inspection and concurrency

`Requests() []stream.CallOptions` returns independent snapshots in capture order.
`ResetRequests()` clears history without changing rules. Unmatched requests are
captured; invalid and pre-canceled requests are not. Concurrent capture order is
mutex acquisition order. Predicates run outside the history lock and must make
their own shared state concurrency-safe.

Configuration, requests, predicate inputs, returned history, and response payloads
are copied through GoAI's JSON representations. Data must be JSON-serializable;
values inside `any` use `encoding/json` decoded types, including `float64` for
numbers. Contexts and tool execution functions retain their identity. Callers must
not modify inputs concurrently with `Stream` copying them.

`Stream` honors its context and `CallOptions.AbortSignal`. Cancellation before
setup returns the context error; cancellation during streaming closes the channel
promptly. An in-flight event, including a finish event, can still be delivered if
its send wins a select against cancellation. Drain the channel or cancel the
context to release the producer. Check the context when consuming a canceled
stream directly rather than treating a finish event as proof it was not canceled.

`ID()` returns the configured ID; `Provider()` returns `"mock"`. Construction and
streaming perform no credential loading or network access.

## Custom streaming

`MockConfig.Stream` takes precedence over rules, responses, and default. It runs
outside the model lock and receives an isolated request snapshot. The model records
the request before invoking the hook, including calls whose hook returns an error.
The hook's context is canceled by the call context or `AbortSignal`, and is released
when forwarding finishes or setup fails. Hooks own their source channel and must
honor that context during setup and production; forwarding closes promptly on
cancellation even when the source channel is idle. Setup errors retain their
identity, and a nil channel without an error is rejected with a clear error.
In-flight calls retain the captured hook across `Configure`. Hook closures must
synchronize their own mutable state and own any event payloads they emit.

## Test fixtures

Use a package-local helper accepting `testing.TB` and `MockConfig`; call
`testutil.NewMockModel(config)` and report construction errors with `t.Fatal`.
Supply an explicit fixture `ID`. Assert against values from `Requests()`.

```go
// Empty stream, single script, and a two-call script sequence:
empty := testutil.MockConfig{ID: "empty", Default: &testutil.MockResponse{Events: []stream.Event{}}}
single := testutil.MockConfig{ID: "single", Default: &testutil.MockResponse{Events: script}}
sequence := testutil.MockConfig{ID: "sequence", Responses: []testutil.MockResponse{
    {Events: first},
    {Events: second},
}}
```

`MockTextResponse`, `MockStreamedTextResponse`, `MockToolCallResponse`,
`MockTextWithToolCallResponse`, `MockErrorResponse`, `MockReasoningResponse`, and
`MockUsage` provide event fixtures. Tool-call helper inputs must be JSON-serializable;
invalid inputs panic. Use explicit `Events` for malformed argument fixtures.
