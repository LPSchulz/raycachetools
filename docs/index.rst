:tocdepth: 3

*******************************************************************************
:mod:`raycachetools` --- Ray-distributed memoizing collections and decorators
*******************************************************************************

.. module:: raycachetools

``raycachetools`` extends `cachetools <https://cachetools.readthedocs.io/>`_
by providing **Ray-backed cache implementations** for workers in a Ray
cluster.  Each cache class in this module is a distributed counterpart to a
common ``cachetools`` cache class.  The eviction policies (FIFO, LFU, LRU, RR,
TTL, TLRU) are delegated to ``cachetools`` inside a named Ray actor, while the
public API focuses on the mapping and decorator operations that are useful
across Ray workers.

The cache is **node-local** rather than cluster-global: each Ray node has its
own detached actor for a given cache name, and workers on that node reconnect
to that local actor.

If you are not yet familiar with the cache algorithms and decorator API
provided by ``cachetools``, please read the
`cachetools documentation <https://cachetools.readthedocs.io/>`_ first.  This
document only covers the differences and additions introduced by
``raycachetools``.


Quick start: shared LRU cache across workers on one node
========================================================

The example below shows multiple Ray workers using the same node-local
``RayLRUCache`` via a shared logical cache name.  Once a value has been stored,
later calls for the same key on the same node can reuse it.

.. code-block:: python

   import time
   import ray
   from raycachetools import RayLRUCache, cached

   ray.init(namespace="my-app")  # all processes must use the same namespace

   CACHE_NAME = "example-shared-lru"

   @ray.remote
   def worker(x):
      cache = RayLRUCache(maxsize=128, name=CACHE_NAME)

      @cached(cache=cache)
      def expensive_square(n):
         time.sleep(0.2)  # stand-in for expensive work
         return n * n

      return expensive_square(x)

   inputs = [2, 2, 3, 3, 2]
   results = ray.get([worker.remote(i) for i in inputs])
   print(results)  # [4, 4, 9, 9, 4]

.. important::

   Create the cache handle inside the remote function or Ray actor method that
   uses it.  Do **not** create a cache handle on the main driver and pass that
   handle into remote work, because that binds the remote worker to the
   driver's node-local cache actor rather than the worker's local one.

Direct cache usage (without decorators)
---------------------------------------

You can also use ``RayLRUCache`` directly from workers:

.. code-block:: python

   import time
   import ray
   from raycachetools import RayLRUCache

   ray.init(namespace="my-app")

   CACHE_NAME = "example-direct-lru"

   @ray.remote
   def worker(x):
      cache = RayLRUCache(maxsize=128, name=CACHE_NAME)
      if x in cache:
         return cache[x]

      time.sleep(0.2)  # stand-in for expensive work
      value = x * x
      cache[x] = value
      return value

   inputs = [5, 5, 6, 6, 5]
   results = ray.get([worker.remote(i) for i in inputs])
   print(results)  # [25, 25, 36, 36, 25]


How it works
============

Each ``Ray*Cache`` instance spawns (or reconnects to) a **named, detached**
Ray actor called :class:`CacheActor` on the **current Ray node**.  The actor
name is derived from the machine hostname and the user-supplied cache name, so
the same logical cache name resolves to a different actor on each node.  The
actor holds a regular
``cachetools`` cache internally, but stores Ray ``ObjectRef`` handles rather
than raw values:

* **Writes** — the caller sends the raw value to the actor, which places it
  in the Ray object store with :func:`ray.put`.  Because the *actor* calls
  ``ray.put``, the actor is the **owner** of the resulting ``ObjectRef`` —
  ensuring the object remains alive as long as the detached actor exists,
  regardless of the calling worker's lifetime.
* **Reads** — the caller fetches the ``ObjectRef`` from the actor and
  resolves it locally via :func:`ray.get`.

Because cache access and mutation are serialised inside a single actor, **no
local locking is required to protect the cache object itself**.  This is the
key architectural difference from ``cachetools``, where callers must supply
their own ``lock`` or ``condition`` objects for thread-safe cache access.

This does not make decorated functions single-flight.  If multiple workers
miss the same key at the same time, they may all compute the value and then
race to store it.  The actor keeps cache state consistent, but
``raycachetools`` does not currently provide the equivalent of the
``cachetools.cached(..., condition=...)`` stampede-mitigation mechanism.

.. note::

   Ray must be initialised (``ray.init()``) before creating any
   ``raycachetools`` cache.

.. important::

   Named actors in Ray are scoped to a **namespace**.  If you do not pass an
   explicit ``namespace`` to :func:`ray.init`, each process receives its own
   anonymous namespace — meaning each process will create a **separate**
   per-node detached actor even when using the same cache ``name``.

   To reconnect to the same per-node cache across multiple drivers or jobs on
   the same node, always specify a namespace:

   .. code-block:: python

      ray.init(address="auto", namespace="my-app")

   All processes that should share the same caches must use the same
   namespace.


Key differences from cachetools
===============================

+-------------------------------+------------------------------------------+----------------------------------------+
| Feature                       | ``cachetools``                           | ``raycachetools``                      |
+===============================+==========================================+========================================+
| Process scope                 | In-process only                          | Shared by workers on the same Ray node |
+-------------------------------+------------------------------------------+----------------------------------------+
| Required ``name`` parameter   | N/A                                      | Every cache requires a unique ``name`` |
+-------------------------------+------------------------------------------+----------------------------------------+
| Thread safety                 | User provides ``lock`` / ``condition``   | Serialised by the Ray actor — no       |
|                               |                                          | locking needed                         |
+-------------------------------+------------------------------------------+----------------------------------------+
| ``getsizeof`` support         | Supported                                | Not supported (size = number of items) |
+-------------------------------+------------------------------------------+----------------------------------------+
| ``lock`` / ``condition``      | Accepted by decorators                   | Not accepted; actor handles cache      |
|                               |                                          | state synchronisation                  |
+-------------------------------+------------------------------------------+----------------------------------------+
| Concurrent misses             | ``condition`` can prevent duplicate work | Duplicate work is possible; the actor  |
|                               | for the same cache key                   | only serialises cache access           |
+-------------------------------+------------------------------------------+----------------------------------------+
| Policy-specific APIs          | Exposes methods/properties such as       | Only the common mapping surface and    |
|                               | ``expire()``, ``timer``, ``ttl``,        | documented additions are exposed       |
|                               | ``ttu`` and ``choice`` where applicable  |                                        |
+-------------------------------+------------------------------------------+----------------------------------------+
| Cache lifetime                | Follows Python object lifetime           | Detached actor survives the creator    |
|                               |                                          | on that node for the lifetime of the   |
|                               |                                          | Ray cluster (object store), or until   |
|                               |                                          | :meth:`destroy` is called              |
+-------------------------------+------------------------------------------+----------------------------------------+


Cache implementations
=====================

All distributed cache classes inherit from the internal
:class:`_RayCacheBase`, which implements the
:class:`collections.abc.MutableMapping` interface and provides
:attr:`maxsize`, :attr:`currsize`, :attr:`name`, :attr:`actor_name`, and
:attr:`hostname` properties.

The constructors intentionally do not accept the ``getsizeof`` argument from
``cachetools``.  ``currsize`` is therefore the number of cached entries, not a
user-defined aggregate object size.

.. class:: RayCache(maxsize, name)

   Ray-backed counterpart to :class:`cachetools.Cache`.

   :param int maxsize: Maximum number of cache entries.
   :param str name: Unique Ray actor name.

.. class:: RayFIFOCache(maxsize, name)

   Distributed First In First Out (FIFO) cache.
   Ray-backed counterpart to :class:`cachetools.FIFOCache`.

   :param int maxsize: Maximum number of cache entries.
   :param str name: Unique Ray actor name.

.. class:: RayLFUCache(maxsize, name)

   Distributed Least Frequently Used (LFU) cache.
   Ray-backed counterpart to :class:`cachetools.LFUCache`.

   :param int maxsize: Maximum number of cache entries.
   :param str name: Unique Ray actor name.

.. class:: RayLRUCache(maxsize, name)

   Distributed Least Recently Used (LRU) cache.
   Ray-backed counterpart to :class:`cachetools.LRUCache`.

   :param int maxsize: Maximum number of cache entries.
   :param str name: Unique Ray actor name.

.. class:: RayRRCache(maxsize, name, choice=random.choice)

   Distributed Random Replacement (RR) cache.
   Ray-backed counterpart to :class:`cachetools.RRCache`.

   :param int maxsize: Maximum number of cache entries.
   :param str name: Unique Ray actor name.
   :param choice: A callable that selects an arbitrary element from a
      non-empty sequence (default: :func:`random.choice`).

.. class:: RayTTLCache(maxsize, ttl, name, timer=time.monotonic)

   Distributed LRU cache with per-item time-to-live.
   Ray-backed counterpart to :class:`cachetools.TTLCache`.

   :param int maxsize: Maximum number of cache entries.
   :param ttl: Time-to-live for each item (in seconds by default).
   :param str name: Unique Ray actor name.
   :param timer: A callable returning the current time (default:
      :func:`time.monotonic`).

.. class:: RayTLRUCache(maxsize, ttu, name, timer=time.monotonic)

   Distributed Time-aware Least Recently Used (TLRU) cache.
   Ray-backed counterpart to :class:`cachetools.TLRUCache`.

   :param int maxsize: Maximum number of cache entries.
   :param ttu: A callable ``ttu(key, value, now)`` returning the
      expiration time for a cache item.
   :param str name: Unique Ray actor name.
   :param timer: A callable returning the current time (default:
      :func:`time.monotonic`).


Common properties and methods
-----------------------------

All ``Ray*Cache`` classes expose the following:

.. attribute:: name

   The logical cache name supplied by the user (read-only).

.. attribute:: actor_name

   The concrete node-local Ray actor name backing this cache (read-only).

.. attribute:: hostname

   The hostname of the node that owns this cache actor (read-only).

.. attribute:: maxsize

   The maximum size of the cache (read-only).

.. attribute:: currsize

   The current number of items in the cache (read-only).

.. method:: destroy()

   Kill the backing Ray actor and release all cached object references.
   After calling :meth:`destroy`, the cache instance is no longer usable.

Cache data and actor state therefore live for the lifetime of the active
Ray cluster (object store), unless explicitly removed earlier via
:meth:`destroy`.

The full :class:`collections.abc.MutableMapping` interface (``__getitem__``,
``__setitem__``, ``__delitem__``, ``__contains__``, ``__iter__``,
``__len__``, ``get``, ``pop``, ``setdefault``, ``popitem``, ``clear``) is
also available.


Reconnecting to an existing cache
----------------------------------

Because the backing actor is **detached**, creating a new proxy with the same
``name`` on the same node will reconnect to the existing node-local actor
rather than spawning a new one.  This allows multiple workers (or successive
script invocations) on the same node to share the same local cache:

.. code-block:: python

   import ray
   from raycachetools import RayLRUCache

   ray.init(namespace="my-app")  # same namespace on every process

   # Worker A
   cache = RayLRUCache(maxsize=1024, name="shared-lru")
   cache["key"] = expensive_result

   # Worker B (same node, possibly a different process)
   cache = RayLRUCache(maxsize=1024, name="shared-lru")
   print(cache["key"])  # returns the value stored by Worker A


Memoizing decorators
====================

``raycachetools`` provides :func:`cached` and :func:`cachedmethod`
decorators with the same basic cache/key/info shape as ``cachetools`` and the
following differences:

* **No** ``lock`` **or** ``condition`` **parameters** — synchronisation is
  handled by the Ray actor.
* The ``cache`` argument should typically be a ``Ray*Cache`` instance.

.. decorator:: cached(cache, key=raycachetools.keys.hashkey, info=False)

   Decorator to wrap a function with a memoizing callable that saves results
   in a node-local distributed cache.

   Works like :func:`cachetools.cached` for the common cache/key/info use
   case, except that ``lock`` and ``condition`` are not accepted.  Since the
   backing Ray actor serialises cache access, no local locking is needed for
   cache state consistency.

   Concurrent cache misses for the same key may still execute the wrapped
   function more than once.  Use an application-level single-flight mechanism
   if duplicate work is unacceptable.

   The decorator exposes the following attributes on the wrapped function:

   * ``cache`` — the cache object.
   * ``cache_key`` — the key function.
   * ``cache_clear()`` — clear the cache.
   * ``cache_info()`` — *(only when* ``info=True`` *)* return a
     ``CacheInfo(hits, misses, maxsize, currsize)`` named tuple.

   .. note::

      Hit and miss counters are tracked **per-worker** (locally), not
      globally across the node.

   Example:

   .. code-block:: python

      import ray
      from raycachetools import RayLRUCache, cached

      ray.init(namespace="my-app")

      @cached(cache=RayLRUCache(maxsize=256, name="fib-cache"), info=True)
      def fib(n):
          return n if n < 2 else fib(n - 1) + fib(n - 2)

      print(fib(42))           # 267914296
      print(fib.cache_info())  # CacheInfo(hits=..., misses=..., maxsize=256, currsize=...)

.. decorator:: cachedmethod(cache, key=raycachetools.keys.methodkey, info=False)

   Decorator to wrap an instance method with a memoizing callable.

   As with :func:`cachetools.cachedmethod`, ``cache`` is a **callable** that
   receives ``self`` and typically returns a mutable mapping (for example a
   ``Ray*Cache``). Returning ``None`` is also supported and disables caching
   for that instance.

   ``lock`` and ``condition`` are not accepted.

   .. note::

      With ``info=True``, :meth:`cache_info` requires a mapping-like cache.
      If ``cache(self)`` returns ``None``, calling ``cache_info()`` raises
      :class:`TypeError`.

   Example:

   .. code-block:: python

      import ray
      from raycachetools import RayLRUCache, cachedmethod

      ray.init(namespace="my-app")

      class MyService:
          def __init__(self):
              self._cache = RayLRUCache(maxsize=128, name="svc-cache")

          @cachedmethod(lambda self: self._cache, info=True)
          def compute(self, x):
              return x ** 2

      svc = MyService()
      svc.compute(7)                # 49
      print(svc.compute.cache_info())


*************************************************************************
:mod:`raycachetools.keys` --- Key functions for memoizing decorators
*************************************************************************

.. module:: raycachetools.keys

This module provides the same key functions as :mod:`cachetools.keys`:

.. function:: hashkey(*args, **kwargs)

   Return a cache key for the specified hashable arguments.

.. function:: methodkey(self, *args, **kwargs)

   Like :func:`hashkey`, but ignores the first positional argument
   (``self``).

.. function:: typedkey(*args, **kwargs)

   Like :func:`hashkey`, but arguments of different types produce distinct
   cache keys (e.g. ``typedkey(3)`` ≠ ``typedkey(3.0)``).

.. function:: typedmethodkey(self, *args, **kwargs)

   Like :func:`typedkey`, but ignores the first positional argument
   (``self``).

These functions are serialisable by Ray, making them safe to use in
distributed settings.


***********************************************************************************
:mod:`raycachetools.func` --- :func:`functools.lru_cache` compatible decorators
***********************************************************************************

.. module:: raycachetools.func

Ray-backed counterparts for the :mod:`cachetools.func` decorators.  Each
decorator automatically creates (or reconnects to) a named **node-local** Ray
actor.

All decorators accept an additional ``name`` parameter:

:param str name:
   Explicit actor name.  When ``None`` (the default), a deterministic name
   is derived from the decorated function's module and qualified name.
   Re-decorating the same function with ``name=None`` therefore reconnects to
   the same underlying actor-backed cache.

``maxsize=None`` is accepted for API familiarity, but is currently implemented
as a very large ``cachetools`` cache size (``2**31``).  Treat it as
practically unbounded for normal use, not as a mathematically unlimited cache.

.. decorator:: fifo_cache(maxsize=128, typed=False, name=None)

   Distributed FIFO memoizing decorator.
   Also supports bare usage as ``@fifo_cache``.

.. decorator:: lfu_cache(maxsize=128, typed=False, name=None)

   Distributed LFU memoizing decorator.
   Also supports bare usage as ``@lfu_cache``.

.. decorator:: lru_cache(maxsize=128, typed=False, name=None)

   Distributed LRU memoizing decorator.
   Also supports bare usage as ``@lru_cache``.

.. decorator:: rr_cache(maxsize=128, choice=random.choice, typed=False, name=None)

   Distributed Random Replacement memoizing decorator.
   Also supports bare usage as ``@rr_cache``.

.. decorator:: ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False, name=None)

   Distributed TTL + LRU memoizing decorator.  By default the time-to-live
   is 600 seconds.
   Also supports bare usage as ``@ttl_cache``.

Example:

.. code-block:: python

   import ray
   from raycachetools.func import lru_cache

   ray.init(namespace="my-app")

   @lru_cache(maxsize=256)
   def count_vowels(sentence):
       return sum(sentence.casefold().count(v) for v in "aeiou")

   count_vowels("Hello, World!")  # 3
   print(count_vowels.cache_info())

.. note::

   Unlike the ``cachetools.func`` decorators, these are **not**
   thread-safe by local locking — they rely on the Ray actor for
   serialisation instead.  The ``name`` parameter is unique to
   ``raycachetools.func``.  When used from remote tasks or actors, create the
   decorated function inside the remote context so it binds to the local
   node's cache actor rather than a handle captured from the main driver.



*******************************************************************
:mod:`raycachetools._actor` --- Internal cache actor implementation
*******************************************************************

.. module:: raycachetools._actor

.. class:: CacheActor(cache_cls, cache_args=(), cache_kwargs=None)

   A named, detached Ray actor that manages a ``cachetools`` cache whose
   values are Ray ``ObjectRef`` handles.  This is an implementation detail;
   users interact with the ``Ray*Cache`` proxy classes instead.

   The actor tracks hit/miss statistics internally and exposes a
   ``cache_info()`` method returning ``(hits, misses, maxsize, currsize)``.
