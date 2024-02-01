<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/edgenai/edgen">
    <!-- TODO: uncomment to show discord -->
    <!-- <img alt="Discord" src="https://img.shields.io/discord/1163068604074426408?logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FMMUcgBtV"> -->
</p>

<h3 align="center">
    A Local GenAI API Server: A drop-in replacement for OpenAI's API
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
    <img src="docs/assets/edgen_architecture_overview.svg" alt="⚡Edgen architecture overview" height="500">
    <p align="center">⚡Edgen architecture overview</p>
</div>

- [x] **Optimized Inference**: You don't need to take a PhD in AI optimization. ⚡Edgen abstracts the complexity of optimizing inference for different hardware, platforms and models.
- [x] **Modular**: ⚡Edgen is **model** and **runtime** agnostic. New models can be added easily and ⚡Edgen can select the best runtime for the user's hardware: you don't need to keep up about the latest models and ML runtimes - **⚡Edgen will do that for you**.
- [x] **Model Caching**: ⚡Edgen caches foundational models locally, so 1 model can power hundreds of different apps - users don't need to download the same model multiple times.
- [x] **OpenAI Compliant API**: ⚡Edgen is a drop-in replacement for OpenAI.

⚡Edgen lets you use GenAI in your app, completely **locally** on your user's devices, for **free** and with **data-privacy**. It's a drop-in replacement for OpenAI (it uses the a compatible API), supports various functions like text and image generation, speech-to-text, and text-to-speech, and works on Windows, Linux, and MacOS.

### Endpoints

- [x] \[Chat\] [Completions](https://docs.edgen.co/api-reference/chat)
- [x] \[Audio\] [Transcriptions](https://docs.edgen.co/api-reference/audio)
- [ ] \[Image\] Generation
- [ ] \[Chat\] Multimodal chat completions
- [ ] \[Audio\] Speech

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

1. [Download](https://edgen.co/download) ⚡Edgen
2. Chat with ⚡[EdgenChat](https://chat.edgen.co)

# Developers

The following sections are for people looking to contribute to ⚡Edgen.

## Architecture Overview

## Quickstart

Edgen uses [Nix](https://nixos.org/) for dependency management and development environments.
To get up-and-running quickly, [install Nix](https://nixos.org/download.html),
[enable Nix flakes](https://nixos.wiki/wiki/Flakes), and run:

```bash
nix develop
# TODO: Insert run command here
```

Then open your favorite IDE from the shell, and you're ready to go!

## Communication Channels

- [Edgen Discord server](https://discord.gg/QUXbwqdMRs): Real time discussions with the ⚡Edgen team and other users.
- [GitHub issues](https://github.com/edgenai/edgen/issues): Feature requests, bugs.
- [GitHub discussions](https://github.com/edgenai/edgen/discussions/): Q&A.
- [Blog](https://blog.edgen.co): Big announcements.

## Special Thanks

- [`llama.cpp`](https://github.com/ggerganov/llama.cpp/tree/master),
  [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp), and [`ggml`](https://github.com/ggerganov/ggml) for being
  an excellent getting-on point for this space.
