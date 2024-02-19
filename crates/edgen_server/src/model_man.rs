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

//! Model Manager Endpoints

use std::fmt;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, SystemTimeError};

use axum::extract;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use thiserror;
use tracing::warn;
use utoipa::ToSchema;

use edgen_core::settings;

/// GET `/v1/models`: returns a list of model descriptors for all models in all model directories.
///
/// For any error, the endpoint returns "internal server error".
pub async fn list_models() -> Response {
    match list_all_models().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => internal_server_error(&format!("model manager: cannot list models: {:?}", e)),
    }
}

/// GET `/v1/models{:id}`: returns the model descriptor for the model indicated by 'id'.
///
/// For any error, the endpoint returns "internal server error".
pub async fn retrieve_model(extract::Path(id): extract::Path<String>) -> Response {
    match model_id_to_desc(&id).await {
        Ok(d) => Json(d).into_response(),
        Err(e) => {
            internal_server_error(&format!("model manager: cannot get model {}: {:?}", id, e))
        }
    }
}

/// DELETE `/v1/models{:id}`: deletes the model indicated by 'id'.
///
/// For any error, the endpoint returns "internal server error".
pub async fn delete_model(extract::Path(id): extract::Path<String>) -> Response {
    match remove_model(&id).await {
        Ok(d) => Json(d).into_response(),
        Err(e) => internal_server_error(&format!(
            "model manager: cannot delete model {}: {:?}",
            id, e
        )),
    }
}

fn internal_server_error(msg: &str) -> Response {
    warn!("{}", msg);
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}

/// Model Descriptor
#[derive(ToSchema, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct ModelDesc {
    /// model Id
    pub id: String,
    /// when the file was created
    pub created: u64,
    /// object type, always 'model'
    pub object: String,
    /// repo owner
    pub owned_by: String,
}

/// Model Deletion Status
#[derive(ToSchema, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct ModelDeletionStatus {
    /// model Id
    pub id: String,
    /// object type, always 'model'
    pub object: String,
    /// repo owner
    pub deleted: bool,
}

#[derive(Debug, thiserror::Error)]
enum PathError {
    Generic(String),
    ModelNotFound,
    ParseError(#[from] ParseError),
    IOError(#[from] std::io::Error),
    TimeError(#[from] SystemTimeError),
}

impl Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

async fn list_all_models() -> Result<Vec<ModelDesc>, PathError> {
    let completions_dir = settings::chat_completions_dir().await;
    let transcriptions_dir = settings::audio_transcriptions_dir().await;

    let mut v = vec![];

    list_models_in_dir(Path::new(&completions_dir), &mut v).await?;
    list_models_in_dir(Path::new(&transcriptions_dir), &mut v).await?;
    Ok(v)
}

async fn list_models_in_dir(path: &Path, v: &mut Vec<ModelDesc>) -> Result<(), PathError> {
    let es = tokio::fs::read_dir(path).await;
    if es.is_err() {
        warn!("model manager: cannot read directory {:?} ({:?})", path, es);
        return Err(PathError::IOError(es.unwrap_err()));
    };
    let mut es = es.unwrap();
    loop {
        let e = es.next_entry().await;
        if e.is_err() {
            warn!("model manager: cannot get entry: {:?}", e);
            break;
        }
        let tmp = e.unwrap();
        if tmp.is_none() {
            break;
        }
        let tmp = tmp.unwrap();
        match path_to_model_desc(tmp.path().as_path()).await {
            Ok(m) => v.push(m),
            Err(e) => {
                warn!(
                    "model manager: invalid entry in directory {:?}: {:?}",
                    path, e
                );
            }
        }
    }
    Ok(())
}

async fn model_id_to_desc(id: &str) -> Result<ModelDesc, PathError> {
    let path = search_model(id).await?;
    path_to_model_desc(path.as_path()).await
}

async fn search_model(id: &str) -> Result<PathBuf, PathError> {
    let model = model_id_to_path(id)?;
    let dir = settings::chat_completions_dir().await;
    let path = Path::new(&dir).join(&model);
    if path.is_dir() {
        return Ok(path);
    }
    let dir = settings::audio_transcriptions_dir().await;
    let path = Path::new(&dir).join(&model);
    if path.is_dir() {
        return Ok(path);
    }
    Err(PathError::ModelNotFound)
}

async fn remove_model(id: &str) -> Result<ModelDeletionStatus, PathError> {
    let model = search_model(id).await?;
    let _ = tokio::fs::remove_dir_all(model).await?;
    Ok(ModelDeletionStatus {
        id: id.to_string(),
        object: "model".to_string(),
        deleted: true,
    })
}

async fn path_to_model_desc(path: &Path) -> Result<ModelDesc, PathError> {
    let f = path
        .file_name()
        .ok_or(PathError::Generic("empty path".to_string()))?;
    let model = f
        .to_str()
        .ok_or(PathError::Generic("invalid file name".to_string()))?;
    let (owner, repo) = parse_path(model)?;
    let metadata = tokio::fs::metadata(path).await?;
    if !metadata.is_dir() {
        return Err(PathError::Generic("not a directory".to_string()));
    };
    let tp = match metadata.created() {
        Ok(n) => n,
        Err(_) => SystemTime::UNIX_EPOCH, // unknown
    };

    let created = tp.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();

    Ok(ModelDesc {
        id: to_model_id(&owner, &repo),
        created: created,
        object: "model".to_string(),
        owned_by: owner.to_string(),
    })
}

fn to_model_id(owner: &str, repo: &str) -> String {
    format!("{}/{}", owner, repo)
}

fn model_id_to_path(id: &str) -> Result<PathBuf, ParseError> {
    let (owner, repo) = parse_model_id(id)?;
    let s = format!("models--{}--{}", owner, repo);
    Ok(PathBuf::from(s))
}

#[derive(Debug, PartialEq, thiserror::Error)]
enum ParseError {
    MissingSeparator,
    NotaModel,
    NoOwner,
    NoRepo,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

fn parse_model_id(id: &str) -> Result<(String, String), ParseError> {
    let vs = id.split("/").collect::<Vec<&str>>();
    if vs.len() < 2 {
        return Err(ParseError::MissingSeparator);
    }

    let owner = vs[0].to_string();
    if owner.is_empty() {
        return Err(ParseError::NoOwner);
    }

    let repo = if vs.len() > 2 {
        vs[1..].join("/")
    } else {
        vs[1].to_string()
    };
    if repo.is_empty() {
        return Err(ParseError::NoRepo);
    }

    Ok((owner, repo))
}

fn parse_path(model_string: &str) -> Result<(String, String), ParseError> {
    let vs = model_string.split("--").collect::<Vec<&str>>();

    if vs.len() < 3 {
        return Err(ParseError::MissingSeparator);
    }

    if vs[0] != "models" {
        return Err(ParseError::NotaModel);
    }

    // the owner is always the second
    // if the original owner contained double dashes
    // we won't find him
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
    use std::time::SystemTime;

    use tempfile;

    // --- Parse Model Id -------------------------------------------------------------------------
    #[test]
    fn parse_simple_model_id_valid() {
        assert_eq!(
            parse_model_id("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_model_id_slashes_in_repo() {
        assert_eq!(
            parse_model_id("TheBloke/TinyLlama/1.1B/Chat/v1.0-GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama/1.1B/Chat/v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_model_id_slashes_in_owner_valid() {
        assert_eq!(
            parse_model_id("The/Bloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Ok((
                "The".to_string(),
                "Bloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_model_id_slashes_in_owner_valid() {
        assert_ne!(
            parse_model_id("The/Bloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_model_id_no_slashes_between_owner_and_repo() {
        assert_eq!(
            parse_model_id("The-Bloke-TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::MissingSeparator)
        );
    }

    #[test]
    fn fail_model_id_no_slashes_after_owner() {
        assert_eq!(
            parse_model_id("The-Bloke"),
            Err(ParseError::MissingSeparator)
        );
    }

    #[test]
    fn fail_model_id_no_repo() {
        assert_eq!(parse_model_id("The-Bloke/"), Err(ParseError::NoRepo));
    }

    #[test]
    fn fail_model_id_no_owner() {
        assert_eq!(
            parse_model_id("/The-Bloke-TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::NoOwner)
        );
    }

    #[test]
    fn fail_model_id_nothing() {
        assert_eq!(parse_model_id("/"), Err(ParseError::NoOwner));
    }

    #[test]
    fn fail_model_id_even_less() {
        assert_eq!(parse_model_id(""), Err(ParseError::MissingSeparator));
    }

    // --- Parse Model Entry ----------------------------------------------------------------------
    #[test]
    fn parse_path_simple_valid() {
        assert_eq!(
            parse_path("models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_path_dashes_in_repo_valid() {
        assert_eq!(
            parse_path("models--TheBloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn parse_path_dashes_in_owner_valid() {
        assert_eq!(
            parse_path("models--The--Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "The".to_string(),
                "Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_path_dashes_in_owner() {
        assert_ne!(
            parse_path("models--The--Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF"),
            Ok((
                "TheBloke".to_string(),
                "TinyLlama--1.1B--Chat--v1.0--GGUF".to_string(),
            ))
        );
    }

    #[test]
    fn fail_path_does_not_start_with_model() {
        assert_eq!(
            parse_path("datasets--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::NotaModel)
        );
    }

    #[test]
    fn fail_path_no_dashes_between_owner_and_repo() {
        assert_eq!(
            parse_path("models--TheBloke-TinyLlama-1.1B-Chat-v1.0-GGUF"),
            Err(ParseError::MissingSeparator)
        );
    }

    #[test]
    fn fail_path_no_dashes_after_owner() {
        assert_eq!(
            parse_path("models--TheBloke"),
            Err(ParseError::MissingSeparator)
        );
    }

    #[test]
    fn fail_path_no_repo() {
        assert_eq!(parse_path("models--TheBloke--"), Err(ParseError::NoRepo));
    }

    #[test]
    fn fail_path_no_owner() {
        assert_eq!(parse_path("models----"), Err(ParseError::NoOwner));
    }

    #[test]
    fn fail_path_no_model() {
        assert_eq!(
            parse_path("--TheBlock--whatever"),
            Err(ParseError::NotaModel)
        );
    }

    #[test]
    fn fail_path_nothing() {
        assert_eq!(parse_path(""), Err(ParseError::MissingSeparator));
    }

    // --- Roundtrip ------------------------------------------------------------------------------
    #[test]
    fn simple_roundtrip() {
        let paths = vec![
            "models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF",
            "models--The--Bloke--TinyLlama--1.1B--Chat--v1.0--GGUF",
            "models--TheBloke--TinyLlama--1.1B--Chat--v1.0--GGUF",
        ];
        for path in paths.into_iter() {
            let (owner, repo) = parse_path(path).unwrap();
            let id = to_model_id(&owner, &repo);
            let pb = model_id_to_path(&id).unwrap();
            let round = pb.as_path().to_str().unwrap();
            assert_eq!(path, round);
        }
    }

    // --- path to desc ---------------------------------------------------------------------------
    #[tokio::test]
    async fn test_list_models_in_dir() {
        let bloke = "TheBloke";
        let the = "The";
        let r1 = "TinyLlama-1.1B-Chat-v1.0-GGUF";
        let r2 = "Bloke--TinyLlama-1.1B-Chat-v1.0-GGUF";
        let r3 = "TinyLlama--1.1B--Chat--v1.0--GGUF";
        let f1 = format!("models--{}--{}", bloke, r1);
        let f2 = format!("models--{}--{}", the, r2);
        let f3 = format!("models--{}--{}", bloke, r3);
        let f4 = "invisible".to_string();
        let f5 = "models--TheBlokeInvisible".to_string();
        let f6 = "tmp".to_string();

        let temp = tempfile::tempdir().expect("cannot create tempfile");

        let recent = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 2; // careful with leap seconds

        std::fs::create_dir(temp.path().join(&f1)).expect(&format!("cannot create dir {:?}", f1));
        std::fs::create_dir(temp.path().join(&f2)).expect(&format!("cannot create dir {:?}", f2));
        std::fs::create_dir(temp.path().join(&f3)).expect(&format!("cannot create dir {:?}", f3));
        std::fs::create_dir(temp.path().join(&f4)).expect(&format!("cannot create dir {:?}", f4));
        std::fs::create_dir(temp.path().join(&f5)).expect(&format!("cannot create dir {:?}", f5));
        std::fs::create_dir(temp.path().join(&f6)).expect(&format!("cannot create dir {:?}", f6));

        let mut v = vec![];

        let _ = list_models_in_dir(temp.path(), &mut v)
            .await
            .expect("cannot list directory");

        assert_eq!(v.len(), 3);

        println!("recent is {}", recent);
        for m in v {
            assert_eq!(m.object, "model");
            if m.owned_by != the {
                assert_eq!(m.owned_by, bloke);
            }
            if m.id != format!("{}/{}", bloke, r1) && m.id != format!("{}/{}", bloke, r3) {
                assert_eq!(m.id, format!("{}/{}", the, r2));
            }
            println!("{:?}", m);

            let d = m.created.checked_sub(recent).unwrap();
            assert!(d <= 3);
        }
    }
}
