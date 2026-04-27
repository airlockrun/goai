# goai

Go port of the [Vercel AI SDK](https://github.com/vercel/ai) — streaming text generation, tool calling, image/audio generation, embeddings, reranking, transcription.

The TypeScript-flavored API surface is preserved as faithfully as Go syntax allows, so patterns from the Vercel AI SDK docs translate directly. See [NOTICE](NOTICE) for attribution.

## Install

```go
import "github.com/airlockrun/goai"
```

```bash
go get github.com/airlockrun/goai
```

Requires Go 1.26+.

## Streaming text

The main entry point — equivalent to the Vercel AI SDK's `streamText()`:

```go
result, err := goai.StreamText(ctx, stream.Input{
	Model:    yourModel,             // a goai.LanguageModel implementation
	Messages: yourMessages,          // []message.Message
	System:   "You are a helpful assistant.",
	Tools:    yourTools,             // optional
})
if err != nil {
	return err
}

for event := range result.Events() {
	// handle event.TextDelta, event.ToolCall, event.Finish, ...
}
```

`StreamText` automatically loops through tool calls when `MaxSteps > 1`, executing tools and feeding results back into the model — same shape as the upstream multi-step agent loop.

Other top-level functions: `GenerateText`, `GenerateImage`, `GenerateSpeech`, `Transcribe`, `Embed`, `Rerank`.

## Scope

goai tracks vercel/ai upstream. We accept bug fixes specific to the Go port, but not changes that diverge from upstream's logic or API design — if you have an idea that improves the SDK conceptually, take it to [vercel/ai](https://github.com/vercel/ai) first; once it lands upstream, it'll flow into goai naturally. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Companion projects

- [airlock](https://github.com/airlockrun/airlock) (AGPL-3.0) — self-hosted cyborg agent platform
- [agentsdk](https://github.com/airlockrun/agentsdk) (Apache-2.0) — Go SDK for building agents on airlock
- [sol](https://github.com/airlockrun/sol) (Apache-2.0) — agent runtime / CLI utility, built on goai

## License

[Apache-2.0](LICENSE). The Vercel AI SDK is also Apache-2.0; see [NOTICE](NOTICE) for the upstream attribution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). A CLA Assistant bot will prompt you to sign on your first PR (one signature covers all airlockrun projects).

## Security

Email `security@airlock.run`. Do not open public issues for vulnerabilities.
