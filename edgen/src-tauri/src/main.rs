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

#[cfg(not(feature = "no_gui"))]
mod gui;

use edgen_server;
use edgen_server::{cli, start, EdgenResult};

use once_cell::sync::Lazy;

#[cfg(not(feature = "no_gui"))]
fn main() -> EdgenResult {
    Lazy::force(&cli::PARSED_COMMANDS);

    match &cli::PARSED_COMMANDS.subcommand {
        None => serve(&cli::PARSED_COMMANDS, true)?,
        Some(cli::Command::Serve(args)) => serve(&cli::PARSED_COMMANDS, !args.nogui)?,
        Some(_) => start(&cli::PARSED_COMMANDS)?,
    }

    Ok(())
}

#[cfg(feature = "no_gui")]
fn main() -> EdgenResult {
    Lazy::force(&cli::PARSED_COMMANDS);
    start(&cli::PARSED_COMMANDS)
}

#[cfg(not(feature = "no_gui"))]
fn serve(command: &'static cli::TopLevel, start_gui: bool) -> EdgenResult {
    let handle = std::thread::spawn(|| match start(command) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{:?}", e);
            std::process::exit(1);
        }
    });

    if start_gui {
        gui::run();
    }

    handle.join()?
}
