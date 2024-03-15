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

use core::time::Duration;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::executor::block_on;
use tokio::select;
use tokio::sync::{oneshot, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{info, span, Level};

/// An asynchronous `OnceCell` with expiration semantics.
///
/// A `Perishable` resembles an `RwLock<OnceCell<T>>`, but with the added property that the inner
/// value `T` will be dropped after a period of time if not accessed, and must be re-constructed
/// thereafter.
///
/// Creating a `Perishable` spawns a new asynchronous thread that watches the inner value for
/// inactivity.
pub struct Perishable<T> {
    /// The inner state and contents of this [`Perishable`].
    inner: Arc<PerishableInner<T>>,

    /// Channel used to signal that this [`Perishable`] has been dropped.
    _drop_tx: oneshot::Sender<()>,
}

struct PerishableInner<T> {
    current_value: RwLock<Option<T>>,
    active_signal: ActiveSignal,
    state: Arc<RwLock<PerishableState>>,
    perish_callback: RwLock<Option<Box<dyn Fn() + Send + Sync>>>,
}

struct PerishableState {
    active: bool,
    last_accessed: Instant,
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

        let state = Arc::new(RwLock::new(PerishableState {
            active: false,
            last_accessed: Instant::now(),
        }));
        let state_clone0 = state.clone();
        let state_clone1 = state.clone();

        let inner = Arc::new(PerishableInner {
            current_value: RwLock::new(None),
            active_signal: ActiveSignal::new(
                1,
                move || {
                    let mut locked = block_on(state_clone0.write());
                    locked.active = true;
                },
                move || {
                    let mut locked = block_on(state_clone1.write());
                    locked.active = false;
                    locked.last_accessed = Instant::now();
                },
            ),
            state,
            perish_callback: RwLock::new(None),
        });

        let watched_inner = Arc::clone(&inner);

        tokio::spawn(async move {
            let span =
                span!(Level::TRACE, "perishable", "ttl" = ?ttl, "ty" = std::any::type_name::<T>());
            let _ = span.enter();

            let mut drop_rx = drop_rx;
            let watched_inner = watched_inner;

            loop {
                let accessed = {
                    let locked = watched_inner.state.read().await;
                    if locked.active {
                        Instant::now()
                    } else {
                        locked.last_accessed
                    }
                };

                let check_date = if accessed + ttl > Instant::now() {
                    accessed + ttl
                } else {
                    Instant::now() + Duration::from_secs(5)
                };

                select! {
                    _ = &mut drop_rx => break,
                    _ = yield_until(check_date) => {
                        if watched_inner.state.read().await.last_accessed == accessed && watched_inner.current_value.write().await.take().is_some() {
                            info!("A {} has perished", std::any::type_name::<T>());
                            if let Some(callback) = watched_inner.perish_callback.read().await.as_ref() {
                                callback()
                            }
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
    fn touch(&self) -> ActiveSignal {
        self.inner.active_signal.clone()
    }

    /// Returns `true` if the value inside of this wrapper is currently live (i.e., has been
    /// initialized and has not expired).
    pub async fn is_alive(&self) -> bool {
        self.inner.current_value.read().await.is_some()
    }

    /// Forcefully drops the inner value.
    pub async fn kill(&self) {
        if self.inner.current_value.write().await.take().is_some() {
            info!("A {} has been killed", std::any::type_name::<T>());
            if let Some(callback) = self.inner.perish_callback.read().await.as_ref() {
                callback()
            }
        }
    }

    /// Sets an optional called when the value perishes.
    pub async fn set_callback(&self, callback: Option<impl Fn() + Send + Sync + 'static>) {
        let callback = callback.map(|f| Box::new(f) as Box<dyn Fn() + Send + Sync>);
        *self.inner.perish_callback.write().await = callback;
    }

    /// Gets a RAII read guard for the value, possibly initializing it.
    ///
    /// If the value is not initialized, it will be initialized, which may be expensive.
    pub async fn get_or_init<F: AsyncConstructor<T>>(
        &self,
        constructor: F,
    ) -> (ActiveSignal, PerishableReadGuard<'_, T>) {
        let signal = self.touch();

        {
            let guard = self.inner.current_value.read().await;

            if guard.is_some() {
                return (signal, PerishableReadGuard(guard));
            }
        }

        // Value isn't initialized. Acquire a write lock and initialize it.
        let mut guard = self.inner.current_value.write().await;

        info!("(Re)Creating a new {}", std::any::type_name::<T>());
        *guard = Some(constructor.construct().await);

        (signal, PerishableReadGuard(guard.downgrade()))
    }

    /// Gets a RAII read guard for the value, attempting to initialize it and yielding an error
    /// if initialization fails.
    ///
    /// This has the same semantics as [`get_or_init`], but returns a `Result`.
    pub async fn get_or_try_init<E, F: AsyncConstructor<Result<T, E>>>(
        &self,
        constructor: F,
    ) -> Result<(ActiveSignal, PerishableReadGuard<'_, T>), E> {
        let signal = self.touch();

        {
            let guard = self.inner.current_value.read().await;

            if guard.is_some() {
                return Ok((signal, PerishableReadGuard(guard)));
            }
        }

        // Value isn't initialized. Acquire a write lock and initialize it.
        let mut guard = self.inner.current_value.write().await;

        info!("(Re)Creating a new {}", std::any::type_name::<T>());
        *guard = Some(constructor.construct().await?);

        Ok((signal, PerishableReadGuard(guard.downgrade())))
    }

    /// Gets a RAII write guard for the inner value, possibly initializing it.
    ///
    /// If the value is not initialized, it will be initialized, which may be expensive.
    pub async fn get_or_init_mut<F: AsyncConstructor<T>>(
        &self,
        constructor: F,
    ) -> (ActiveSignal, PerishableWriteGuard<'_, T>) {
        let signal = self.touch();
        let mut guard = self.inner.current_value.write().await;

        if guard.is_none() {
            info!("(Re)Creating a new {}", std::any::type_name::<T>());
            *guard = Some(constructor.construct().await);
        }

        (signal, PerishableWriteGuard(guard))
    }

    /// Gets a RAII write guard for the value, attempting to initialize it and yielding an error
    /// if initialization fails.
    ///
    /// This has the same semantics as [`get_or_init_mut`], but returns a `Result`.
    pub async fn get_or_try_init_mut<E, F: AsyncConstructor<Result<T, E>>>(
        &self,
        constructor: F,
    ) -> Result<(ActiveSignal, PerishableWriteGuard<'_, T>), E> {
        let signal = self.touch();
        let mut guard = self.inner.current_value.write().await;

        if guard.is_none() {
            info!("(Re)Creating a new {}", std::any::type_name::<T>());
            *guard = Some(constructor.construct().await?);
        }

        Ok((signal, PerishableWriteGuard(guard)))
    }
}

/// The inner state of an [`ActiveSignal`].
struct ActiveSignalInner {
    /// The current number of references.
    refs: usize,

    /// The threshold used to decide when to call `active_callback` and `inactive_callback`.
    threshold: usize,

    /// The function to be called when the number of references goes above the `threshold`.
    active_callback: Box<dyn Fn() + Send>,

    /// The function to be called when the number of references goes below the `threshold`.
    inactive_callback: Box<dyn Fn() + Send>,
}

/// An object indicating that some resource associated with this signal is getting used by whatever owns this object.
///
/// It may call a specified function when it goes above a certain number of references, and another
/// specified function when either at or below the same number of references. The intention is to mark whatever is
/// associated with this signal as either active or inactive, depending on how many associated signals are currently
/// instantiated.
pub struct ActiveSignal {
    /// This objects inner state.
    inner: Arc<Mutex<ActiveSignalInner>>,
}

impl ActiveSignal {
    /// Creates a new instance of [`ActiveSignal`].
    ///
    /// ## Arguments
    /// * `threshold` - The threshold used to decide when to call `active_callback` and `inactive_callback`.
    /// * `active_callback` - The function to be called when the number of references goes above the `threshold`.
    /// * `inactive_callback` - The function to be called when the number of references goes below the `threshold`.
    fn new(
        threshold: usize,
        active_callback: impl Fn() + Send + 'static,
        inactive_callback: impl Fn() + Send + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ActiveSignalInner {
                refs: 1,
                threshold,
                active_callback: Box::new(active_callback),
                inactive_callback: Box::new(inactive_callback),
            })),
        }
    }
}

impl Clone for ActiveSignal {
    fn clone(&self) -> Self {
        let clone = self.inner.clone();

        if let Ok(mut locked) = self.inner.lock() {
            if locked.refs == locked.threshold {
                locked.active_callback.deref()();
            }
            locked.refs += 1;
        }

        Self { inner: clone }
    }
}

impl Drop for ActiveSignal {
    fn drop(&mut self) {
        if let Ok(mut locked) = self.inner.lock() {
            locked.refs -= 1;
            if locked.refs == locked.threshold {
                locked.inactive_callback.deref()();
            }
        }
    }
}

/// Yields until a [`std::time`]-based [`Instant`] has elapsed.
pub async fn yield_until(t: Instant) {
    let now = Instant::now();

    if t > now {
        tokio::time::sleep(t - Instant::now()).await;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn perishable_lives() {
        let perishable = Perishable::with_ttl(Duration::from_secs(5));

        perishable.get_or_init(|| async { 0 }).await;
        perishable.get_or_init(|| async { 1 }).await;

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await.1, 0);
    }

    #[tokio::test]
    async fn perishable_dies() {
        let perishable = Perishable::with_ttl(Duration::from_millis(100));

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await.1, 0);
        assert!(perishable.is_alive().await);

        tokio::time::sleep(Duration::from_millis(200)).await;

        assert!(!perishable.is_alive().await);

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await.1, 0);
        assert!(perishable.is_alive().await);
    }

    #[tokio::test]
    async fn perishable_touch() {
        let perishable = Perishable::with_ttl(Duration::from_millis(100));

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await.1, 0);
        assert!(perishable.is_alive().await);
        {
            let (_signal, _value) = perishable.get_or_init(|| async { 0 }).await;

            tokio::time::sleep(Duration::from_millis(200)).await;
            assert!(perishable.is_alive().await);
        }

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!perishable.is_alive().await);

        assert_eq!(*perishable.get_or_init(|| async { 0 }).await.1, 0);
        assert!(perishable.is_alive().await);
    }
}
