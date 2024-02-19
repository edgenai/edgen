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

//! Minor Edgen services like version.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use tracing::error;
use utoipa::ToSchema;

/// Reads the version defined in Cargo.toml at compile time in the format
/// `MAJOR.MINOR.PATCH_BUILD`
#[macro_export]
macro_rules! cargo_crate_version {
    () => {
        env!("CARGO_PKG_VERSION")
    };
}

/// Current Edgend Version.
#[derive(ToSchema, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct Version {
    major: u32,
    minor: u32,
    patch: u32,
    build: String,
}

/// GET `/v1/misc/version`: returns the current version of edgend.
///
/// The version is returned as json value with major, minor and patch as integer
/// and build as string (which may be empty).
/// For any error, the version endpoint returns "internal server error".
#[utoipa::path(
        get,
        path = "/misc/version",
        responses(
            (status = 200, description = "OK", body = Version),
            (status = 500, description = "unexpected internal server error")
        ),
)]
pub async fn edgen_version() -> Response {
    match string_to_version(cargo_crate_version!()) {
        Ok(j) => Json(j).into_response(),
        Err(e) => internal_server_error(&e),
    }
}

fn internal_server_error(msg: &str) -> Response {
    error!("[ERROR] {}", msg);
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}

/// Takes a string of the form "1.0.2-build" and turns it into a Version.
/// Fails if the input is not a valid version string.
pub fn string_to_version(vs: &str) -> Result<Version, String> {
    let v1 = vs.split(".").collect::<Vec<&str>>();

    let l = v1.len();
    if l < 3 {
        return Err("incomplete version number".to_string());
    } else if l > 3 {
        return Err("too many components in version number".to_string());
    }

    let major_str = v1[0];
    let minor_str = v1[1];

    // get build if any
    let tmp = v1.last().unwrap(); // we know it's not empty
    let v2 = tmp.split("-").collect::<Vec<&str>>();
    let build = if v2.len() > 1 {
        v2[1..].join("-")
    } else {
        "".to_string()
    };

    let patch_str = v2[0];

    // parse major, minor and patch from string
    let major = match major_str.parse::<u32>() {
        Ok(n) => n,
        _ => return Err("major is not a number".to_string()),
    };

    let minor = match minor_str.parse::<u32>() {
        Ok(n) => n,
        _ => return Err("minor is not a number".to_string()),
    };

    let patch = match patch_str.parse::<u32>() {
        Ok(n) => n,
        _ => return Err("patch is not a number".to_string()),
    };

    Ok(Version {
        major: major,
        minor: minor,
        patch: patch,
        build: build,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_axum_router() {
        let version_router = Router::new().route("/v1/misc/version", get(edgen_version));

        let server = TestServer::new(version_router).expect("cannot instantiate TestServer");

        let response = server.get("/v1/misc/version").await.json::<Version>();

        let expected =
            string_to_version(cargo_crate_version!()).expect("cannot deserialize version string");

        assert_eq!(response, expected);
    }

    #[test]
    fn with_valid_version_no_build() {
        assert_eq!(
            string_to_version("1.0.1"),
            Ok(Version {
                major: 1,
                minor: 0,
                patch: 1,
                build: "".to_string(),
            })
        )
    }

    #[test]
    fn with_valid_version_simple_build() {
        assert_eq!(
            string_to_version("1.0.1-xyz"),
            Ok(Version {
                major: 1,
                minor: 0,
                patch: 1,
                build: "xyz".to_string(),
            })
        )
    }

    #[test]
    fn with_valid_version_dash_in_build() {
        assert_eq!(
            string_to_version("1.0.1-86_64-special-patch"),
            Ok(Version {
                major: 1,
                minor: 0,
                patch: 1,
                build: "86_64-special-patch".to_string(),
            })
        )
    }

    #[test]
    fn with_incomplete_version() {
        assert_eq!(
            string_to_version("1.0"),
            Err("incomplete version number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_major_nan() {
        assert_eq!(
            string_to_version("t.0.1"),
            Err("major is not a number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_minor_nan() {
        assert_eq!(
            string_to_version("1.x.1"),
            Err("minor is not a number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_patch_nan() {
        assert_eq!(
            string_to_version("1.0.x"),
            Err("patch is not a number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_build_no_dash() {
        assert_eq!(
            string_to_version("1.0.1_patch"),
            Err("patch is not a number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_build_wrong() {
        assert_eq!(
            string_to_version("1.0.1.my-patch"),
            Err("too many components in version number".to_string())
        )
    }

    #[test]
    fn with_invalid_version_build_wrong_with_number() {
        assert_eq!(
            string_to_version("1.0.1.2"),
            Err("too many components in version number".to_string())
        )
    }
}
