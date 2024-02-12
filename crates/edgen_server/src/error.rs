//! Edgen Error Handling

use std::any::Any;
use std::error;
use std::io;
use std::str;

use edgen_core::settings;
use edgen_core::llm;
use edgen_core::whisper;

/// Abstraction over all errors that we can handle in edgen.
/// This allows using '?' error handling everywhere for all known error types.
#[derive(Debug)]
pub enum Error {
    /// generic error represented by an error message
    GenericError(String),
    /// error resulting from settings
    SettingsError(settings::SettingsError),
    /// error resulting from the LLM model runtime
    LLMEndpointError(llm::LLMEndpointError),
    /// error resulting from the Whisper model runtime
    WhisperEndpointError(whisper::WhisperEndpointError),
    /// error resulting from an IO error
    IOError(io::Error),
    /// error resulting from invalid UTF-8 encoding
    UTF8Error(str::Utf8Error),
    /// error based on the standard error trait
    StandardError(Box<dyn std::error::Error + Send>),
    /// error for functions returning anything
    AnyError(Box<dyn Any + Send>),
}

impl From<settings::SettingsError> for Error {
    fn from(e: settings::SettingsError) -> Self {
        Error::SettingsError(e)
    }
}

impl From<llm::LLMEndpointError> for Error {
    fn from(e: llm::LLMEndpointError) -> Self {
        Error::LLMEndpointError(e)
    }
}

impl From<whisper::WhisperEndpointError> for Error {
    fn from(e: whisper::WhisperEndpointError) -> Self {
        Error::WhisperEndpointError(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IOError(e)
    }
}

impl From<str::Utf8Error> for Error {
    fn from(e: str::Utf8Error) -> Self {
        Error::UTF8Error(e)
    }
}

impl From<Box<dyn error::Error + Send>> for Error {
    fn from(e: Box<dyn error::Error + Send>) -> Self {
        Error::StandardError(e)
    }
}

impl From<Box<dyn Any + Send>> for Error {
    fn from(e: Box<dyn Any + Send>) -> Self {
        Error::AnyError(e)
    }
}
