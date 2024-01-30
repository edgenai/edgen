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
 
//! Shims around [`smol`][smol], [`tokio`][tokio], and [`glommio`] to provide a unified interface
//! for asynchronous programming.
//!
//! This crate exports [`spawn`], [`spawn_local`], and [`unblock`] functions that will
//! defer to the appropriate runtime depending on the feature flags enabled.
//!
//! The following feature flags are available and **mutually exclusive**:
//!
//! - `runtime-smol`: Use [`smol`] as the runtime.
//! - `runtime-tokio`: Use [`tokio`] as the runtime.
//! - `runtime-glommio`: Use [`glommio`] as the runtime.
//!
//! You must enable **exactly one** of these. Failing to enable any, or enabling more than one,
//! will cause the crate to fail to compile.
//!
//! [smol]: https://docs.rs/smol
//! [tokio]: https://docs.rs/tokio
//! [glommio]: https://docs.rs/glommio

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::future::Future;

static_assertions::assert_cfg!(
    any(
        all(feature = "runtime-smol", not(any(feature = "runtime-tokio", feature = "runtime-glommio"))),
        all(feature = "runtime-tokio", not(any(feature = "runtime-smol", feature = "runtime-glommio"))),
        all(feature = "runtime-glommio", not(any(feature = "runtime-smol", feature = "runtime-tokio"))),
    ),
    "You must enable exactly one of the `runtime-smol`, `runtime-tokio`, or `runtime-glommio` feature flags."
);

/// Spawns a future onto the current executor, causing it to start executing almost immediately.
///
/// This will automatically select for `smol`, `tokio`, or `glommio` depending on the feature
/// flags enabled.
pub async fn spawn<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "runtime-smol")] {
            smol::spawn(future).detach();
        } else if #[cfg(feature = "runtime-tokio")] {
            tokio::spawn(future);
        } else if #[cfg(feature = "runtime-glommio")] {
            glommio::spawn_local(future).detach();
        } else {
            unreachable!("No runtime enabled; build should not have succeeded")
        }
    }
}

/// Spawns a blocking function onto a thread pool, returning a future that resolves when the
/// thread pool has finished executing the function.
pub async fn unblock<F, R>(callback: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "runtime-smol")] {
            smol::unblock(callback).await
        } else if #[cfg(feature = "runtime-tokio")] {
            tokio::task::spawn_blocking(callback).await.unwrap()
        } else if #[cfg(feature = "runtime-glommio")] {
            glommio::executor().spawn_blocking(callback).await
        } else {
            unreachable!("No runtime enabled; build should not have succeeded")
        }
    }
}

/// Blocks the current thread until the provided future completes.
pub fn block_on<F, R>(future: F) -> R
where
    F: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "runtime-smol")] {
            smol::block_on(future)
        } else if #[cfg(feature = "runtime-tokio")] {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(future)
        } else if #[cfg(feature = "runtime-glommio")] {
            // glommio::executor().spawn_local(future)
            todo!()
        } else {
            unreachable!("No runtime enabled; build should not have succeeded")
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_spawn() {
        async fn test() {
            spawn(async {
                println!("Hello, world!");
            })
            .await;
        }

        block_on(test());
    }

    #[test]
    fn test_spawn_blocking() {
        async fn test() {
            unblock(|| {
                println!("Hello, world!");
            })
            .await;
        }

        block_on(test());
    }
}
