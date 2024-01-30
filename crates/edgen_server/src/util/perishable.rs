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
 
//! Lazily self-destructing types.

use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use time::{Duration, OffsetDateTime};
use tokio::select;
use tokio::sync::{oneshot, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{span, Level};

use crate::graceful_shutdown::yield_until;

/// An asynchronous `OnceCell` with expiration semantics.
///
/// A `Perishable` resembles an `RwLock<OnceCell<T>>`, but with the added property that the inner
/// value `T` will be dropped after a period of time if not accessed, and must be re-constructed
/// thereafter.
///
/// Creating a `Perishable` spawns a new asynchronous thread that watches the inner value for
/// inactivity.
pub struct Perishable<T> {
    inner: Arc<Inner<T>>,
    _drop_tx: oneshot::Sender<()>,
}
//TODO add some mechanism that actively touches a perishable, without locking it, for model sessions
struct Inner<T> {
    current_value: RwLock<Option<T>>,
    last_accessed: AtomicI64,
}

/// An asynchronous constructor for a [`Perishable`] value.
///
/// This is implemented, by default, for all `async fn() -> T` types for some `Perishable<T>`.
pub trait AsyncConstructor<T>: Send + Sync + 'static {
    /// Creates a `T` asynchronously.
    fn construct(self) -> Pin<Box<dyn Future<Output = T> + Send + 'static>>;
}

impl<F, Fut, T> AsyncConstructor<T> for F
where
    F: Send + Sync + 'static + FnOnce() -> Fut,
    Fut: Future<Output = T> + Send + 'static,
{
    fn construct(self) -> Pin<Box<dyn Future<Output = T> + Send + 'static>> {
        Box::pin(self())
    }
}

/// A RAII read guard to a [`Perishable`] value.
///
/// This acts similarly to an [`RwLockReadGuard`], and prevents the inner value from being reaped
/// while active
pub struct PerishableReadGuard<'a, T>(RwLockReadGuard<'a, Option<T>>);

impl<'a, T> Deref for PerishableReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // PANIC SAFETY: `PerishableReadGuard` can only be created if the inner `Option<T>` is
        // `Some`.
        self.0.as_ref().unwrap()
    }
}

/// A RAII write guard to a [`Perishable`] value.
///
/// This acts similarly to an [`RwLockWriteGuard`], and prevents the inner value from being reaped
/// while active.
pub struct PerishableWriteGuard<'a, T>(RwLockWriteGuard<'a, Option<T>>);

impl<'a, T> Deref for PerishableWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // PANIC SAFETY: `PerishableWriteGuard` can only be created if the inner `Option<T>` is
        // `Some`, and no other write guards can presently exist to set it to `None`.
        self.0.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for PerishableWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // PANIC SAFETY: `PerishableWriteGuard` can only be created if the inner `Option<T>` is
        // `Some`, and no other write guards can presently exist to set it to `None`.
        self.0.as_mut().unwrap()
    }
}

impl<T> Perishable<T>
where
    T: Send + Sync + 'static,
{
    /// Creates a new, lazily-initialized, perishable value.
    ///
    /// `constructor` will be invoked to create the value if it is not already initialized, or
    /// if it has been intermittently de-initialized after `ttl` seconds of inactivity.
    pub fn with_ttl(ttl: Duration) -> Self {
        let (drop_tx, drop_rx) = oneshot::channel();

        let inner = Arc::new(Inner {
            current_value: RwLock::new(None),
            last_accessed: AtomicI64::new(0),
        });

        let watched_inner = Arc::clone(&inner);

        tokio::spawn(async move {
            let span =
                span!(Level::TRACE, "perishable", "ttl" = ?ttl, "ty" = std::any::type_name::<T>());
            let _ = span.enter();

            let mut drop_rx = drop_rx;
            let watched_inner = watched_inner;

            loop {
                let accessed_unix = watched_inner.last_accessed.load(Ordering::Relaxed);
                let accessed_date = OffsetDateTime::from_unix_timestamp(accessed_unix).unwrap();
                let check_date = if accessed_date + ttl > OffsetDateTime::now_utc() {
                    accessed_date + ttl
                } else {
                    accessed_date + Duration::seconds(5)
                };

                select! {
                    _ = &mut drop_rx => break,
                    _ = yield_until(check_date) => {
                        if watched_inner.last_accessed.load(Ordering::Relaxed) == accessed_unix {
                            let _ = watched_inner.current_value.write().await.take();
                        }
                    }
                }
            }
        });

        Self {
            inner,
            _drop_tx: drop_tx,
        }
    }
}

impl<T: 'static> Perishable<T> {
    /// Indicates that the value has been accessed recently.
    fn touch(&self) {
        self.inner.last_accessed.fetch_max(
            (OffsetDateTime::now_utc() - OffsetDateTime::UNIX_EPOCH).whole_seconds(),
            Ordering::Relaxed,
        );
    }

    /// Returns `true` if the value inside of this wrapper is currently live (i.e., has been
    /// initialized and has not expired).
    pub async fn is_alive(&self) -> bool {
        self.inner.current_value.read().await.is_some()
    }

    /// Gets a RAII read guard for the value, possibly initializing it.
    ///
    /// If the value is not initialized, it will be initialized, which may be expensive.
    pub async fn get_or_init<F: AsyncConstructor<T>>(
        &self,
        constructor: F,
    ) -> PerishableReadGuard<'_, T> {
        self.touch();

        {
            let guard = self.inner.current_value.read().await;

            if guard.is_some() {
                return PerishableReadGuard(guard);
            }
        }

        // Value isn't initialized. Acquire a write lock and initialize it.
        let mut guard = self.inner.current_value.write().await;

        *guard = Some(constructor.construct().await);

        PerishableReadGuard(guard.downgrade())
    }

    /// Gets a RAII read guard for the value, attempting to initialize it and yielding an error
    /// if initialization fails.
    ///
    /// This has the same semantics as [`get_or_init`], but returns a `Result`.
    pub async fn get_or_try_init<E, F: AsyncConstructor<Result<T, E>>>(
        &self,
        constructor: F,
    ) -> Result<PerishableReadGuard<'_, T>, E> {
        self.touch();

        {
            let guard = self.inner.current_value.read().await;

            if guard.is_some() {
                return Ok(PerishableReadGuard(guard));
            }
        }

        // Value isn't initialized. Acquire a write lock and initialize it.
        let mut guard = self.inner.current_value.write().await;

        *guard = Some(constructor.construct().await?);

        Ok(PerishableReadGuard(guard.downgrade()))
    }

    /// Gets a RAII write guard for the inner value, possibly initializing it.
    ///
    /// If the value is not initialized, it will be initialized, which may be expensive.
    pub async fn get_or_init_mut<F: AsyncConstructor<T>>(
        &self,
        constructor: F,
    ) -> PerishableWriteGuard<'_, T> {
        let mut guard = self.inner.current_value.write().await;

        if guard.is_none() {
            *guard = Some(constructor.construct().await);
        }

        PerishableWriteGuard(guard)
    }

    /// Gets a RAII write guard for the value, attempting to initialize it and yielding an error
    /// if initialization fails.
    ///
    /// This has the same semantics as [`get_or_init_mut`], but returns a `Result`.
    pub async fn get_or_try_init_mut<E, F: AsyncConstructor<Result<T, E>>>(
        &self,
        constructor: F,
    ) -> Result<PerishableWriteGuard<'_, T>, E> {
        let mut guard = self.inner.current_value.write().await;

        if guard.is_none() {
            *guard = Some(constructor.construct().await?);
        }

        Ok(PerishableWriteGuard(guard))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn perishable_lives() {
        let perishable = Perishable::with_ttl(Duration::seconds(5));

        perishable.get_or_init(|| async { 0 }).await;
        perishable.get_or_init(|| async { 1 }).await;

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await, 0);
    }

    #[tokio::test]
    async fn perishable_dies() {
        let perishable = Perishable::with_ttl(Duration::seconds(1));

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await, 0);
        assert!(perishable.is_alive().await);

        tokio::time::sleep(Duration::seconds(2).unsigned_abs()).await;

        assert!(!perishable.is_alive().await);

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await, 0);
        assert!(perishable.is_alive().await);
    }
}
