/* Copyright 2023- The Binedge, Lda team. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Command Line Interface
use once_cell::sync::Lazy;

/// The parsed command-line arguments provided to this program. Lazily initialized.
///
/// # Exits
///
/// Exits if [`argh::from_env`] would exit--for example, if the `--help` flag was provided or the
/// program arguments could not be parsed.
pub static PARSED_COMMANDS: Lazy<TopLevel> = Lazy::new(argh::from_env);

/// Toplevel CLI commands and options.
/// Subcommands are optional.
/// If no command is provided "serve" will be invoked with default options.
#[derive(argh::FromArgs, PartialEq, Debug)]
pub struct TopLevel {
    /// subcommands
    #[argh(subcommand)]
    pub subcommand: Option<Command>,
}

/// Subcommands
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
pub enum Command {
    /// starts the server.
    Serve(Serve),

    /// configuration-related subcommands.
    Config(Config),

    /// prints the edgen version to stdout.
    Version(Version),

    /// generates the openapi spec and exit.
    Oasgen(Oasgen),
}

/// Starts the edgen server. This is the default command when no command is provided.
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "serve")]
pub struct Serve {
    /// if present, one or more URIs/hosts to bind the server to. `unix://` (on Linux), `http://`,
    /// and `ws://` are supported.
    /// For use in scripts, it is recommended to explicitly add this option
    /// to make your scripts future-proof.
    #[argh(option, short = 'b')]
    pub uri: Vec<String>,
    /// if present, edgen will not start the GUI;
    /// the default behavior is to start the GUI.
    #[argh(switch, short = 'g')]
    pub nogui: bool,
}

impl Default for Serve {
    fn default() -> Serve {
        Serve {
            uri: Vec::default(),
            nogui: false,
        }
    }
}

/// Configuration-related subcommands.
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "config")]
pub struct Config {
    /// config subcommands
    #[argh(subcommand)]
    pub subcommand: ConfigCommand,
}

/// Configuration-related subcommands.
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
pub enum ConfigCommand {
    /// resets the configuration file to the default settings
    Reset(Reset),
}

/// Resets the configuration to the default settings
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "reset")]
pub struct Reset {}

/// Prints the edgen version to stdout.
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "version")]
pub struct Version {}

/// Generates the Edgen OpenAPI specification.
#[derive(argh::FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "oasgen")]
pub struct Oasgen {
    /// if present, edgen will generate the OpenAPI spec in yaml format;
    /// this is the default and can be omitted.
    /// For use in scripts, it is recommended to use the flag to make your scripts future-proof.
    #[argh(switch, short = 'y')]
    pub yaml: bool,
    /// if present, edgen will generate the OpenAPI spec in JSON format;
    /// the default behavior is to generate yaml output.
    #[argh(switch, short = 'j')]
    pub json: bool,
}
