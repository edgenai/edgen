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
    /// and `ws://` are supported, e.g.:
    /// `edgen -b http://127.0.0.1:3000 -b http://192.168.1.1:3000`.
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

#[cfg(test)]
#[rustfmt::skip]
mod test {
    use super::*;
    use argh::FromArgs;

    #[test]
    fn version() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["version"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Version(Version{}))
            }
        );
    }

    #[test]
    fn config_reset() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["config", "reset"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Config(Config {
                    subcommand: ConfigCommand::Reset(Reset {})
                }))
            }
        );
    }

    #[test]
    fn oasgen_only() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["oasgen"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Oasgen(Oasgen{
                    yaml: false,
                    json: false,
                }))
            }
        );
    }

    #[test]
    fn oasgen_yaml_short() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["oasgen", "-y"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Oasgen(Oasgen{
                    yaml: true,
                    json: false,
                }))
            }
        );
    }

    #[test]
    fn oasgen_yaml_long() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["oasgen", "--yaml"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Oasgen(Oasgen{
                    yaml: true,
                    json: false,
                }))
            }
        );
    }

    #[test]
    fn oasgen_json_short() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["oasgen", "-j"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Oasgen(Oasgen{
                    yaml: false,
                    json: true,
                }))
            }
        );
    }

    #[test]
    fn oasgen_json_long() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["oasgen", "--json"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Oasgen(Oasgen{
                    yaml: false,
                    json: true,
                }))
            }
        );
    }

    #[test]
    fn serve_only() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["serve"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Serve(Serve {
                    uri: [].to_vec(),
                    nogui: false,
                }))
            }
        );
    }

    #[test]
    fn serve_nogui() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["serve", "--nogui"]).expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Serve(Serve {
                    uri: [].to_vec(),
                    nogui: true,
                }))
            }
        );
    }

    #[test]
    fn serve_one_uri() {
        assert_eq!(
            TopLevel::from_args(&["edgen"], &["serve", "--uri", "http://localhost"])
                .expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Serve(Serve {
                    uri: ["http://localhost".to_string()].to_vec(),
                    nogui: false,
                }))
            }
        );
    }

    #[test]
    fn serve_many_uris() {
        assert_eq!(
            TopLevel::from_args(
                &["edgen"],
                &[
                    "serve",
                    "--uri", "http://localhost",
                    "-b", "http://remotehost",
                    "-b", "http://anotherhost",
                    "-b", "http://172.0.0.1:3000",
                    "-b", "http://192.168.5.10",
                ]
            )
            .expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Serve(Serve {
                    uri: [
                        "http://localhost",
                        "http://remotehost",
                        "http://anotherhost",
                        "http://172.0.0.1:3000",
                        "http://192.168.5.10",
                    ]
                    .map(|x| x.to_string())
                    .to_vec(),
                    nogui: false,
                }))
            }
        );
    }

    #[test]
    fn serve_many_uris_nogui() {
        assert_eq!(
            TopLevel::from_args(
                &["edgen"],
                &[
                    "serve",
                    "--nogui",
                    "--uri", "http://localhost",
                    "-b", "http://remotehost",
                    "-b", "http://anotherhost",
                    "-b", "http://172.0.0.1:3000",
                    "-b", "http://192.168.5.10",
                ]
            )
            .expect("from_args failed"),
            TopLevel {
                subcommand: Some(Command::Serve(Serve {
                    uri: [
                        "http://localhost",
                        "http://remotehost",
                        "http://anotherhost",
                        "http://172.0.0.1:3000",
                        "http://192.168.5.10",
                    ]
                    .map(|x| x.to_string())
                    .to_vec(),
                    nogui: true,
                }))
            }
        );
    }
}
