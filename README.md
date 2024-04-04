<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/edgenai/edgen">
    <!-- TODO: uncomment to show discord -->
    <!-- <img alt="Discord" src="https://img.shields.io/discord/1163068604074426408?logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FMMUcgBtV"> -->
</p>

<h3 align="center">
    A Local GenAI API Server: A drop-in replacement for OpenAI's API for Local GenAI
</h3>

<p align="center">
    |
    <!-- TODO: add proper links -->
    <a href="https://docs.edgen.co"><b>Documentation</b></a> |
    <a href="https://blog.edgen.co"><b>Blog</b></a> |
    <a href="https://discord.gg/QUXbwqdMRs"><b>Discord</b></a> |
    <a href="https://github.com/orgs/edgenai/projects/1/views/1"><b>Roadmap</b></a> |
</p>

<div align="center">
    <img src="https://edgen.co/images/demo.gif" alt="EdgenChat, a local chat app powered by ⚡Edgen">
    <p align="center">
        <a href="https://chat.edgen.co">EdgenChat</a>, a local chat app powered by ⚡Edgen
    </p>
</div>

- [x] **OpenAI Compliant API**: ⚡Edgen implements an [OpenAI compatible API](https://docs.edgen.co/api-reference), making it a drop-in replacement.
- [x] **Multi-Endpoint Support**: ⚡Edgen exposes multiple AI endpoints such as chat completions (LLMs) and speech-to-text (Whisper) for audio transcriptions.
- [x] **Model Agnostic**: LLMs (Llama2, Mistral, Mixtral...), Speech-to-text (whisper) and [many others](https://docs.edgen.co/documentation/models).
- [x] **Optimized Inference**: You don't need to take a PhD in AI optimization. ⚡Edgen abstracts the complexity of optimizing inference for different hardware, platforms and models.
- [x] **Modular**: ⚡Edgen is **model** and **runtime** agnostic. New models can be added easily and ⚡Edgen can select the best runtime for the user's hardware: you don't need to keep up about the latest models and ML runtimes - **⚡Edgen will do that for you**.
- [x] **Model Caching**: ⚡Edgen caches foundational models locally, so 1 model can power hundreds of different apps - users don't need to download the same model multiple times.
- [x] **Native**: ⚡Edgen is built in 🦀Rust and is natively compiled to all popular platforms: **Windows, MacOS and Linux**. No docker required.
- [ ] **Graphical Interface**: A graphical user interface to help users efficiently manage their models, endpoints and permissions.

⚡Edgen lets you use GenAI in your app, completely **locally** on your user's devices, for **free** and with **data-privacy**. It's a drop-in replacement for OpenAI (it uses the a compatible API), supports various functions like text generation, speech-to-text and works on Windows, Linux, and MacOS.

### Features

- [x] Session Caching: ⚡Edgen maintains top performance with big contexts (big chat histories), by caching sessions. Sessions are auto-detected in function of the chat history.
- [x] [GPU support](https://github.com/edgenai/edgen#gpu-support): CUDA, Vulkan. Metal

### Endpoints

- [x] \[Chat\] [Completions](https://docs.edgen.co/api-reference/chat)
- [x] \[Audio\] [Transcriptions](https://docs.edgen.co/api-reference/audio)
- [x] \[Embeddings\] [Embeddings](https://platform.openai.com/docs/api-reference/embeddings)
- [ ] \[Image\] Generation
- [ ] \[Chat\] Multimodal chat completions
- [ ] \[Audio\] Speech

### Supported Models

Check in the [documentation](https://docs.edgen.co/documentation/models)

### Supported platforms

- [x] Windows
- [x] Linux
- [x] MacOS

## 🔥 Hot Topics

## Why local GenAI?

- **Data Private**: On-device inference means **users' data** never leave their devices.

- **Scalable**: More and more users? No need to increment cloud computing infrastructure. Just let your users use their own hardware.

- **Reliable**: No internet, no downtime, no rate limits, no API keys.

- **Free**: It runs locally on hardware the user already owns.

## Quickstart

1. [Download](https://edgen.co/download) and start ⚡Edgen
2. Chat with ⚡[EdgenChat](https://chat.edgen.co)

Ready to start your own GenAI application? [Checkout our guides](https://docs.edgen.co/guides)!

⚡Edgen usage:

```
Usage: edgen [<command>] [<args>]

Toplevel CLI commands and options. Subcommands are optional. If no command is provided "serve" will be invoked with default options.

Options:
  --help            display usage information

Commands:
  serve             Starts the edgen server. This is the default command when no
                    command is provided.
  config            Configuration-related subcommands.
  version           Prints the edgen version to stdout.
  oasgen            Generates the Edgen OpenAPI specification.
```

`edgen serve` usage:

```
Usage: edgen serve [-b <uri...>] [-g]

Starts the edgen server. This is the default command when no command is provided.

Options:
  -b, --uri         if present, one or more URIs/hosts to bind the server to.
                    `unix://` (on Linux), `http://`, and `ws://` are supported.
                    For use in scripts, it is recommended to explicitly add this
                    option to make your scripts future-proof.
  -g, --nogui       if present, edgen will not start the GUI; the default
                    behavior is to start the GUI.
  --help            display usage information
```

## GPU Support

⚡Edgen also supports compilation and execution on a GPU, when building from source, through Vulkan, CUDA and Metal.
The following cargo features enable the GPU:

- ~~`llama_vulkan` - execute LLM models using Vulkan. Requires a Vulkan SDK to be installed.~~
- `llama_cuda` - execute LLM models using CUDA. Requires a CUDA Toolkit to be installed.
- ~~`llama_metal` - execute LLM models using Metal.~~
- `whisper_cuda` - execute Whisper models using CUDA. Requires a CUDA Toolkit to be installed.

(Vulkan and Metal related features are currently disabled due to lack of support for memory management)

Note that, at the moment, `llama_vulkan`, `llama_cuda` and `llama_metal` cannot be enabled at the same time.

Example usage (building from source, [you need to first install the prerequisites](https://docs.edgen.co/documentation/getting-started)):

```
cargo run --features llama_vulkan --release -- serve
```

## Architecture Overview

<div align="center">
    <img src="docs/assets/edgen_architecture_overview.svg" alt="⚡Edgen architecture overview" width="400">
    <p align="center">⚡Edgen architecture overview</p>
</div>

## Contribute

If you don't know where to start, check [Edgen's roadmap](https://github.com/orgs/edgenai/projects/1/views/1)!
Before you start working on something, see if there's an existing issue/pull-request. Pop into Discord to check with the team or see if someone's already tackling it.

## Communication Channels

- [Edgen Discord server](https://discord.gg/QUXbwqdMRs): Real time discussions with the ⚡Edgen team and other users.
- [GitHub issues](https://github.com/edgenai/edgen/issues): Feature requests, bugs.
- [GitHub discussions](https://github.com/edgenai/edgen/discussions/): Q&A.
- [Blog](https://blog.edgen.co): Big announcements.

## Special Thanks

- [`llama.cpp`](https://github.com/ggerganov/llama.cpp/tree/master),
  [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp), and [`ggml`](https://github.com/ggerganov/ggml) for being
  an excellent getting-on point for this space.
