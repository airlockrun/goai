# goai

Go port of the [Vercel AI SDK](https://github.com/vercel/ai) — streaming text generation, tool calling, image/audio generation, embeddings, reranking, transcription.

The TypeScript-flavored API surface is preserved as faithfully as Go syntax allows, so patterns from the Vercel AI SDK docs translate directly. See [NOTICE](NOTICE) for attribution.

> [!WARNING]
> **Alpha software.** This is an early-stage port — bugs are likely, especially in code paths that haven't been exercised yet. We use it ourselves but expect to find edge cases. Please [open an issue](https://github.com/airlockrun/goai/issues) for anything that breaks.

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

Streams that close without content or a finish event return an error, including
streams containing only start events and raw metadata. Hosted tool calls carry
`ProviderExecuted: true`; core preserves their results and provider metadata in
assistant-message replay and never executes or refines them locally.

Step usage preserves the provider's `Usage.Raw`. Normalized token counts are
nonnegative; aggregate usage sums reported counts, keeps entirely unreported
counts nil, and omits `Raw` because raw provider payloads cannot be summed.

## Files

Files are an optional provider capability (`provider.FilesProvider`). OpenAI,
Anthropic, Google, DeepSeek, and xAI expose `Files()` for uploads. OpenAI and xAI
also support metadata, downloads, and deletion; the other providers expose only
upload, matching the reference SDK. Unsupported operations return an error that
matches `errors.ErrUnsupported` from the GoAI errors package.

```go
files := openai.New(provider.Options{APIKey: apiKey}).Files()
uploaded, err := goai.UploadFile(ctx, goai.UploadFileInput{
	Files:     files,
	Data:      []byte("Hello from GoAI"),
	MediaType: "text/plain",
	Filename:  "hello.txt",
})
if err != nil {
	return err
}

// References can be replayed directly in a file message part.
part := message.FilePart{
	Data:     message.FileDataReference{Reference: uploaded.ProviderReference},
	MimeType: uploaded.MediaType,
	Filename: uploaded.Filename,
}
_ = part

download, err := goai.DownloadFile(ctx, goai.FileInput{
	Files: files,
	File:  uploaded.ProviderReference,
})
if err != nil {
	return err
}
defer download.Content.Close()
```

`GetFileMetadata` and `DeleteFile` accept the same `FileInput`. Upload accepts
exactly one of `Data`, `DataReader`, or `DataBase64`. Omitted media types are
detected from inline bytes; readers default to `application/octet-stream`.
Callers own upload readers
and must close download streams. OpenAI and xAI stream multipart uploads without
buffering the file. Google buffers uploads for its resumable protocol and polls
until processing completes; DeepSeek buffers at most 64 MiB plus one byte for
validation. Context cancellation applies to HTTP requests and Google polling.

File `ProviderOptions` are namespaced, for example
`map[string]any{"openai": openai.FilesOptions{Purpose: "user_data"}}`.
OpenAI supports `purpose` and `expiresAfter`; xAI supports `teamId` and
`expiresAfter`; DeepSeek supports `expiresAfter`; Google supports `displayName`,
`pollIntervalMs`, and `pollTimeoutMs`. Google warns when `Filename` is supplied.

## Provider APIs

Provider implementations are checked against Vercel AI SDK `ai@7.0.93`.
OpenAI, Azure, Hugging Face, and xAI default language models use the Responses
protocol. Azure and xAI expose explicit `Chat` constructors; Hugging Face exposes
`TextGeneration` for its inference text-generation endpoint. The
`provider/openresponses` package supports generic Responses-compatible endpoints.
Cohere chat uses the v2 API. Google image generation uses Gemini
`generateContent`; retired Imagen IDs are rejected.

Additional modality support includes:

- Voyage embeddings and reranking (`provider/voyage`).
- Bedrock reranking and Together AI embeddings, images, and reranking.
- Baseten, Fireworks, and Perplexity embeddings; DeepInfra images.
- Google, Vertex, Mistral, and xAI unary speech and transcription.
- ElevenLabs transcription and fal speech and transcription.

Baseten embeddings require a model URL ending in `/sync` or `/sync/v1`.
OpenAI diarized transcription is available through `TranscribeDetailed`, which
preserves string segment IDs and speaker labels beyond the shared transcription
segment contract. Realtime transcription, video generation, durable batch, and
speech translation are outside this API surface.

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
