//! Edgen Error Handling

use std::any::Any;
use std::error;
use std::fmt;
use std::fmt::Display;
use std::io;
use std::str;

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
