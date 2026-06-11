"""Ray-distributed memoizing collections and decorators."""

__all__ = (
    "RayCache",
    "RayFIFOCache",
    "RayLFUCache",
    "RayLRUCache",
    "RayRRCache",
    "RayTLRUCache",
    "RayTTLCache",
    "cached",
    "cachedmethod",
)

__version__ = "0.1.0"

import collections
import collections.abc
import random
import re
import time

import cachetools
import ray

from . import keys
from ._actor import CacheActor, _Missing

_CacheInfo = collections.namedtuple(
    "CacheInfo", ["hits", "misses", "maxsize", "currsize"]
)


def _sanitize_hostname(hostname):
    """Return a hostname-safe prefix for Ray actor names."""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", hostname)


def _get_local_node_info():
    """Return ``(hostname, node_ip)`` for the current Ray worker process."""
    node_id = ray.get_runtime_context().get_node_id()
    for node in ray.nodes():
        if node.get("NodeID") != node_id:
            continue
        hostname = (
            node.get("NodeManagerHostname")
            or node.get("NodeName")
            or node.get("NodeManagerAddress")
        )
        node_ip = node.get("NodeManagerAddress")
        if hostname is None or node_ip is None:
            break
        return hostname, node_ip
    raise RuntimeError("Could not resolve the current Ray node's hostname")


class _RayCacheBase(collections.abc.MutableMapping):
    """Base proxy that presents a ``MutableMapping`` interface backed by a
    node-local named Ray :class:`CacheActor`.

    On write the proxy sends the raw value to the actor, which calls
    :func:`ray.put` internally so that the **actor** owns the resulting
    ``ObjectRef``.  On read the proxy fetches the ``ObjectRef`` from the
    actor and resolves it locally via :func:`ray.get`.
    """

    def __init__(self, cache_cls, cache_args, cache_kwargs, name):
        self._name = name
        self._hostname, self._node_ip = _get_local_node_info()
        self._actor_name = f"{_sanitize_hostname(self._hostname)}_{name}"
        try:
            self._actor = ray.get_actor(self._actor_name)
        except ValueError:
            try:
                self._actor = CacheActor.options(
                    name=self._actor_name,
                    lifetime="detached",
                    num_cpus=0,
                    max_concurrency=16,
                    resources={f"node:{self._node_ip}": 0.001},
                ).remote(cache_cls, cache_args, cache_kwargs)
            except ray.exceptions.ActorAlreadyExistsError:
                self._actor = ray.get_actor(self._actor_name)

    # -- MutableMapping interface --------------------------------------------

    def __getitem__(self, key):
        result = ray.get(self._actor.get.remote(key))
        if isinstance(result, _Missing):
            raise KeyError(key)
        return ray.get(result.ref)

    def __setitem__(self, key, value):
        ray.get(self._actor.put.remote(key, value))

    def __delitem__(self, key):
        try:
            ray.get(self._actor.delete.remote(key))
        except ray.exceptions.RayTaskError as exc:
            if isinstance(exc.cause, KeyError):
                raise KeyError(key) from None
            raise

    def __contains__(self, key):
        return ray.get(self._actor.contains.remote(key))

    def __iter__(self):
        return iter(ray.get(self._actor.keys.remote()))

    def __len__(self):
        return ray.get(self._actor.len.remote())

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"maxsize={self.maxsize!r}, currsize={self.currsize!r})"
        )

    # -- dict-like helpers ---------------------------------------------------

    def get(self, key, default=None):
        result = ray.get(self._actor.get.remote(key))
        if isinstance(result, _Missing):
            return default
        return ray.get(result.ref)

    def pop(self, key, *args):
        try:
            result = ray.get(self._actor.pop.remote(key))
        except ray.exceptions.RayTaskError as exc:
            if isinstance(exc.cause, KeyError):
                if args:
                    return args[0]
                raise KeyError(key) from None
            raise
        return ray.get(result.ref)

    def setdefault(self, key, default=None):
        wrapper = ray.get(self._actor.setdefault.remote(key, default))
        return ray.get(wrapper.ref)

    def popitem(self):
        try:
            key, wrapper = ray.get(self._actor.popitem.remote())
        except ray.exceptions.RayTaskError as exc:
            if isinstance(exc.cause, KeyError):
                raise KeyError(f"{type(self).__name__} is empty") from None
            raise
        return key, ray.get(wrapper.ref)

    def clear(self):
        ray.get(self._actor.clear.remote())

    # -- batch operations ----------------------------------------------------

    def get_many(self, keys):
        """Return a dict ``{key: value}`` for all *keys* found in the cache.

        Keys not present in the cache are omitted from the result.
        Uses a single actor round-trip.
        """
        results = ray.get(self._actor.get_many.remote(keys))
        return {
            k: ray.get(v.ref) for k, v in results.items() if not isinstance(v, _Missing)
        }

    def put_many(self, items):
        """Insert multiple ``(key, value)`` pairs in a single actor call.

        *items* is an iterable of ``(key, value)`` pairs.
        """
        ray.get(self._actor.put_many.remote(list(items)))

    def delete_many(self, keys):
        """Delete multiple keys in a single actor call.

        Keys not present in the cache are silently skipped.
        """
        ray.get(self._actor.delete_many.remote(list(keys)))

    # -- properties ----------------------------------------------------------

    @property
    def name(self):
        """The logical cache name supplied by the user."""
        return self._name

    @property
    def actor_name(self):
        """The concrete node-local Ray actor name backing this cache."""
        return self._actor_name

    @property
    def hostname(self):
        """The hostname of the node that owns this cache actor."""
        return self._hostname

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return ray.get(self._actor.get_maxsize.remote())

    @property
    def currsize(self):
        """The current size of the cache."""
        return ray.get(self._actor.get_currsize.remote())

    # -- lifecycle -----------------------------------------------------------

    def destroy(self):
        """Kill the backing actor and release all cached object references."""
        ray.kill(self._actor)


# ---------------------------------------------------------------------------
# Concrete cache types — mirror the cachetools hierarchy
# ---------------------------------------------------------------------------


class RayCache(_RayCacheBase):
    """Distributed equivalent of :class:`cachetools.Cache`.

    Eviction must be managed manually (or by a subclass).
    """

    def __init__(self, maxsize, name):
        super().__init__(
            cache_cls=cachetools.Cache,
            cache_args=(maxsize,),
            cache_kwargs={},
            name=name,
        )


class RayFIFOCache(_RayCacheBase):
    """Distributed First In First Out (FIFO) cache."""

    def __init__(self, maxsize, name):
        super().__init__(
            cache_cls=cachetools.FIFOCache,
            cache_args=(maxsize,),
            cache_kwargs={},
            name=name,
        )


class RayLFUCache(_RayCacheBase):
    """Distributed Least Frequently Used (LFU) cache."""

    def __init__(self, maxsize, name):
        super().__init__(
            cache_cls=cachetools.LFUCache,
            cache_args=(maxsize,),
            cache_kwargs={},
            name=name,
        )


class RayLRUCache(_RayCacheBase):
    """Distributed Least Recently Used (LRU) cache."""

    def __init__(self, maxsize, name):
        super().__init__(
            cache_cls=cachetools.LRUCache,
            cache_args=(maxsize,),
            cache_kwargs={},
            name=name,
        )


class RayRRCache(_RayCacheBase):
    """Distributed Random Replacement (RR) cache."""

    def __init__(self, maxsize, name, choice=random.choice):
        super().__init__(
            cache_cls=cachetools.RRCache,
            cache_args=(maxsize,),
            cache_kwargs={"choice": choice},
            name=name,
        )


class RayTTLCache(_RayCacheBase):
    """Distributed LRU cache with per-item time-to-live (TTL)."""

    def __init__(self, maxsize, ttl, name, timer=time.monotonic):
        super().__init__(
            cache_cls=cachetools.TTLCache,
            cache_args=(maxsize, ttl),
            cache_kwargs={"timer": timer},
            name=name,
        )


class RayTLRUCache(_RayCacheBase):
    """Distributed Time-aware Least Recently Used (TLRU) cache."""

    def __init__(self, maxsize, ttu, name, timer=time.monotonic):
        super().__init__(
            cache_cls=cachetools.TLRUCache,
            cache_args=(maxsize, ttu),
            cache_kwargs={"timer": timer},
            name=name,
        )


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------


def cached(cache, key=keys.hashkey, info=False):
    """Decorator to wrap a function with a memoizing callable that saves
    results in a :class:`_RayCacheBase` (or any ``MutableMapping``).

    Since the backing Ray actor serialises access, no local locking is
    required.
    """
    from ._cached import _wrapper

    def decorator(func):
        if info:
            if isinstance(cache, _RayCacheBase):

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, cache.maxsize, cache.currsize)
            elif isinstance(cache, collections.abc.Mapping):

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, None, len(cache))
            else:

                def make_info(hits, misses):
                    return _CacheInfo(hits, misses, 0, 0)

            return _wrapper(func, cache, key, info=make_info)
        else:
            return _wrapper(func, cache, key)

    return decorator


def cachedmethod(cache, key=keys.methodkey, info=False):
    """Decorator to wrap a method with a memoizing callable that saves
    results in a :class:`_RayCacheBase` (or any ``MutableMapping``).
    """
    from ._cachedmethod import _wrapper

    def decorator(method):
        if info:

            def make_info(c, hits, misses):
                if isinstance(c, _RayCacheBase):
                    return _CacheInfo(hits, misses, c.maxsize, c.currsize)
                elif isinstance(c, collections.abc.Mapping):
                    return _CacheInfo(hits, misses, None, len(c))
                else:
                    raise TypeError("cache(self) must return a mutable mapping")

            return _wrapper(method, cache, key, info=make_info)
        else:
            return _wrapper(method, cache, key)

    return decorator
