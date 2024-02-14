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

//! Edgen Error Handling

use std::any::Any;
use std::error;
use std::fmt;
use std::fmt::Display;
use std::io;
use std::str;
use std::string;

use thiserror;

use edgen_core::llm;
use edgen_core::settings;
use edgen_core::whisper;

use crate::model;

/// Abstraction over all errors that we can handle in edgen.
/// This allows using '?' error handling everywhere for all known error types.
#[derive(Debug, thiserror::Error)]
pub enum EdgenError {
    /// generic error represented by an error message
    GenericError(String),
    /// error resulting from an error constructing a model struct
    ModelError(#[from] model::ModelError),
    /// error resulting from settings
    SettingsError(#[from] settings::SettingsError),
    /// error resulting from the LLM model runtime
    LLMEndpointError(#[from] llm::LLMEndpointError),
    /// error resulting from the Whisper model runtime
    WhisperEndpointError(#[from] whisper::WhisperEndpointError),
    /// error resulting from an IO error
    IOError(#[from] io::Error),
    /// error resulting from invalid UTF-8 encoding
    UTF8Error(#[from] str::Utf8Error),
    /// error resulting from invalid UTF-8 encoding
    FromUTF8Error(#[from] string::FromUtf8Error),
    /// error resulting from tokio::JoinError
    JoinError(#[from] tokio::task::JoinError),
    /// error based on the standard error trait
    StandardError(Box<dyn std::error::Error + Send>),
    /// error for functions returning anything
    AnyError(Box<dyn Any + Send>),
}

impl From<Box<dyn error::Error + Send>> for EdgenError {
    fn from(e: Box<dyn error::Error + Send>) -> Self {
        EdgenError::StandardError(e)
    }
}

impl From<Box<dyn Any + Send>> for EdgenError {
    fn from(e: Box<dyn Any + Send>) -> Self {
        EdgenError::AnyError(e)
    }
}

impl Display for EdgenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::ModelError;
    use edgen_core::llm::LLMEndpointError;
    use edgen_core::settings::SettingsError;
    use edgen_core::whisper::WhisperEndpointError;
    use std::fs;

    // ----------------------------------------------------
    // Model Error
    // ----------------------------------------------------
    fn gen_edgen_model_error() -> Result<(), EdgenError> {
        let _ = gen_model_error()?;
        Ok(())
    }

    fn gen_model_error() -> Result<(), ModelError> {
        Err(ModelError::NotPreloaded)
    }

    #[test]
    fn from_model_error() {
        assert_eq!(
            format!("{:?}", gen_edgen_model_error()),
            "Err(ModelError(NotPreloaded))".to_string(),
        );
    }

    // ----------------------------------------------------
    // Settings Error
    // ----------------------------------------------------
    fn gen_edgen_settings_error() -> Result<(), EdgenError> {
        let _ = gen_settings_error()?;
        Ok(())
    }

    fn gen_settings_error() -> Result<(), SettingsError> {
        Err(SettingsError::AlreadyInitialised)
    }

    #[test]
    fn from_settings_error() {
        assert_eq!(
            format!("{:?}", gen_edgen_settings_error()),
            "Err(SettingsError(AlreadyInitialised))".to_string(),
        );
    }

    // ----------------------------------------------------
    // LLM Error
    // ----------------------------------------------------
    fn gen_edgen_llm_error() -> Result<(), EdgenError> {
        let _ = gen_llm_error()?;
        Ok(())
    }

    fn gen_llm_error() -> Result<(), LLMEndpointError> {
        Err(LLMEndpointError::Load("does not exist".to_string()))
    }

    #[test]
    fn from_llm_error() {
        assert_eq!(
            format!("{:?}", gen_edgen_llm_error()),
            "Err(LLMEndpointError(Load(\"does not exist\")))".to_string(),
        );
    }

    // ----------------------------------------------------
    // Whisper Error
    // ----------------------------------------------------
    fn gen_edgen_whisper_error() -> Result<(), EdgenError> {
        let _ = gen_whisper_error()?;
        Ok(())
    }

    fn gen_whisper_error() -> Result<(), WhisperEndpointError> {
        Err(WhisperEndpointError::Load("does not exist".to_string()))
    }

    #[test]
    fn from_whisper_error() {
        assert_eq!(
            format!("{:?}", gen_edgen_whisper_error()),
            "Err(WhisperEndpointError(Load(\"does not exist\")))".to_string(),
        );
    }

    // ----------------------------------------------------
    // IO Error
    // ----------------------------------------------------
    fn gen_io_error() -> Result<(), EdgenError> {
        let _ = fs::read("/does/certainly/not/exist")?;
        Ok(())
    }

    #[test]
    fn from_io_error() {
        assert_eq!(
            format!("{:?}", gen_io_error()),
            "Err(IOError(Os { code: 2, kind: NotFound, message: \"No such file or directory\" }))"
                .to_string(),
        );
    }

    // ----------------------------------------------------
    // UTF Error
    // ----------------------------------------------------
    fn gen_from_utf8_error() -> Result<(), EdgenError> {
        let _ = String::from_utf8(vec![0, 159])?;
        Ok(())
    }

    #[allow(invalid_from_utf8)]
    fn gen_utf8_error() -> Result<(), EdgenError> {
        let _ = str::from_utf8(&[0, 159])?;
        Ok(())
    }

    #[test]
    fn from_utf_error() {
        assert!(
            format!("{:?}", gen_from_utf8_error()).starts_with("Err(FromUTF8Error(FromUtf8Error {")
        );
        assert!(format!("{:?}", gen_utf8_error()).starts_with("Err(UTF8Error(Utf8Error {"));
    }

    // ----------------------------------------------------
    // Join Error
    // ----------------------------------------------------
    async fn gen_join_error() -> Result<(), EdgenError> {
        let _ = tokio::spawn(async {
            panic!("boom");
        })
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn join_error() {
        assert!(
            format!("{:?}", gen_join_error().await).starts_with("Err(JoinError(JoinError::Panic(")
        );
    }
}
