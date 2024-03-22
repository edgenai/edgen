 # Edgen 
A Local GenAI API Server: A drop-in replacement for OpenAI's API for Local GenAI
- [Description](#description)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installing](#installing)
  - [Executing program](#executing-program)
- [Documentation](#documentation)
- [Help](#help)
- [Running the Application Locally](#running-the-application-locally)
- [License](#license)

## Description

Edgen is a Local, private GenAI server alternative to OpenAI. No GPU is required. Run AI models locally: LLMs (Llama2, Mistral, Mixtral...), Speech-to-text (whisper) and many others.

## Getting Started

### Dependencies

- [Rust](https://www.rust-lang.org/tools/install)
- [NodeJs](https://nodejs.org/en/download/)
- [pnpm](https://pnpm.io/installation)

### Installing

See the [releases](https://github.com/edgenai/edgen/releases) page for the latest binary. All major platforms are supported.


## Documentation
See the [documentation page](https://docs.edgen.co) for help and support 

## Help
Should any error be encountered with the Rust toolchain, the following command will install the required toolchain for the project

```shell
rustup toolchain add beta-2023-11-21 --profile minimal
```

## Running the Application Locally 
To run the application locally, ensure the dependencies are met 
```shell
pnpm install
pnpm tauri dev 
``` 

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](../LICENSE) file for details
      