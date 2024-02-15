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

use axum::response::{IntoResponse, Json, Response};

pub async fn list_models() -> Response {
    Json(true).into_response()
}

pub async fn retrieve_model() -> Response {
    Json(true).into_response()
}

pub async fn delete_model() -> Response {
    Json(true).into_response()
}

#[derive(Debug, PartialEq)]
enum ParseError {
    MissingDashes,
    NotaModel,
    NoOwner,
    NoRepo,
}

fn parse_model_entry(model_string: &str) -> Result<(String, String), ParseError> {
    let vs = model_string.split("--").collect::<Vec<&str>>();

    if vs.len() < 3 {
        return Err(ParseError::MissingDashes);
    }

    if vs[0] != "models" {
        return Err(ParseError::NotaModel);
    }

    // the owner is always the second
    // if the original owner contained double dashes
    // we won't found him
    let owner = vs[1].to_string();
    if owner.is_empty() {
        return Err(ParseError::NoOwner);
    }

    let repo = if vs.len() > 3 {
        vs[2..].join("--")
    } else {
        vs[2].to_string()
    };
    if repo.is_empty() {
        return Err(ParseError::NoRepo);
    }

    Ok((owner, repo))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_simple_valid() {
        assert_eq!(
            parse_model_entry("models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_dashes_in_repo_valid() {
        assert_eq!(
            parse_model_entry("models--TheBloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_dashes_in_owner_valid() {
        assert_eq!(
            parse_model_entry("models--The--Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "The".to_string(),
                "Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_dashes_in_owner() {
        assert_ne!(
            parse_model_entry("models--The--Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_does_not_start_with_model() {
        assert_eq!(
            parse_model_entry("datasets--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::NotaModel)
        );
    }

    #[test]
    fn fail_no_dashes_between_owner_and_repo() {
        assert_eq!(
            parse_model_entry("models--TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::MissingDashes)
        );
    }

    #[test]
    fn fail_no_dashes_after_owner() {
        assert_eq!(
            parse_model_entry("models--TheBloke"),
            Err(ParseError::MissingDashes)
        );
    }

    #[test]
    fn fail_no_repo() {
        assert_eq!(
            parse_model_entry("models--TheBloke--"),
            Err(ParseError::NoRepo)
        );
    }

    #[test]
    fn fail_no_owner() {
        assert_eq!(parse_model_entry("models----"), Err(ParseError::NoOwner));
    }

    #[test]
    fn fail_no_model() {
        assert_eq!(
            parse_model_entry("--TheBlock--whatever"),
            Err(ParseError::NotaModel)
        );
    }
}
