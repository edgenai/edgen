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
 
//! Mechanisms for shutting down application without destroying anything important.

use time::{Duration, OffsetDateTime};
use tokio::signal;
use tokio::sync::OnceCell;
use tracing::warn;

/// The duration between [`global_shutdown_starts`] and [`global_shutdown_ends`].
pub const SHUTDOWN_GRACE_PERIOD: Duration = Duration::seconds(30);

static SHUTDOWN_INVOKED_AT: OnceCell<OffsetDateTime> = OnceCell::const_new();

/// Listens for signals that cause the application to shut down; namely, `CTRL+C`.
async fn signal_listener() -> OffsetDateTime {
    while signal::ctrl_c().await.is_err() { /* spin */ }

    warn!(
        "Global shutdown has been invoked at {}, and will result in a hard termination at {}",
        OffsetDateTime::now_utc(),
        OffsetDateTime::now_utc() + SHUTDOWN_GRACE_PERIOD
    );

    OffsetDateTime::now_utc()
}

/// Resolves when a global shutdown has started.
///
/// All threads **should** start gracefully exiting by this time.
pub async fn global_shutdown_starts() {
    yield_until(*SHUTDOWN_INVOKED_AT.get_or_init(signal_listener).await).await;
}

/// Resolves when the application is about to unconditionally shut down, following
/// [`global_shutdown_starts`].
///
/// This fires after a grace period of [`SHUTDOWN_GRACE_PERIOD`].
pub async fn global_shutdown_ends() {
    yield_until(*SHUTDOWN_INVOKED_AT.get_or_init(signal_listener).await + SHUTDOWN_GRACE_PERIOD)
        .await;
}

/// Yields until a [`time`]-based [`OffsetDateTime`] has elapsed.
pub async fn yield_until(t: OffsetDateTime) {
    let now = OffsetDateTime::now_utc();

    if t > now {
        tokio::time::sleep((t - OffsetDateTime::now_utc()).unsigned_abs()).await;
    }
}
