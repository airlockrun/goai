# OpenResponses

`openresponses.New` provides language models for a Responses-compatible endpoint.
`BaseURL` is required and includes the API prefix, for example `https://example.com/v1`.
Requests use `POST {BaseURL}/responses`. `Model`, `LanguageModel`, and `Responses`
select the same protocol implementation.

```go
p := openresponses.New(openresponses.Options{
    Name:    "local",
    BaseURL: "http://localhost:1234/v1",
    APIKey:  "optional-key",
})
m := p.Model("my-model")
```

The implementation is shared with OpenAI Responses through
`openai.NewResponsesModel` and `openai.ResponsesConfig`. It supports message and
function-tool input, structured output, streamed text and reasoning, tool calls,
usage, and typed stream errors. Generic endpoints do not infer OpenAI reasoning
capabilities from model names. Provider options use the flat
`openresponses.ResponsesOptions` shape; the endpoint must support any requested
extensions. Responses item metadata uses the `openai` namespace for round trips
through the shared message converter.

Per-call headers override configured headers. A configured API key is applied
last as bearer authentication. `HTTPClient` allows a custom transport. Non-language
model factories return nil.

Azure's `Model` and `LanguageModel` use Responses; `Chat` selects its deployment
Chat Completions API. Azure refreshes `TokenProvider` on every request. Its
Responses API version defaults to `v1`; explicit deployment APIs keep their own
version default.

Hugging Face's `Model` and `LanguageModel` use
`https://router.huggingface.co/v1/responses`. `ResponsesBaseURL` configures this
endpoint independently of `BaseURL`, which serves inference modalities.
`TextGeneration` selects prompt-based inference. Image generation and embeddings
remain available through their inference model factories.
