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

//! A brutally simple session management framework for Edgen's transport-agnostic, Protobuf-based
//! message-passing protocol.
//!
//! The heart of this crate is the [`Server`], which can open many [`Session`]s backed by an
//! arbitrary [`EnvelopeHandler`].

extern crate alloc;
extern crate core;

use std::future::Future;
use std::time::Duration;

pub mod llm;
pub mod whisper;

pub mod settings;

pub mod perishable;
pub mod request;

/// A generic [`Box`]ed [`Future`], used to emulate `async` functions in traits.
pub type BoxedFuture<'a, T> = Box<dyn Future<Output = T> + Send + Unpin + 'a>;

/// Return the [`Duration`] that cleanup threads should wait before looking for and freeing unused
/// resources, after last doing so.
pub fn cleanup_interval() -> Duration {
    // TODO this should come from the settings
    Duration::from_secs(20)
}
