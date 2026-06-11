"""Ray actor that manages a cachetools cache storing ObjectRef handles."""

__all__ = ("CacheActor", "ObjRefWrapper", "MISSING")

import threading

import ray


class ObjRefWrapper:
    """Wraps a Ray ObjectRef to prevent automatic resolution in actor calls.

    When an ObjectRef is passed as a top-level argument to a Ray remote
    method, Ray automatically resolves it to the underlying value.  Wrapping
    it in this container prevents that behaviour so the actor receives the
    reference itself rather than the materialised object.
    """

    __slots__ = ("ref",)

    def __init__(self, ref: "ray.ObjectRef"):
        self.ref = ref


class _Missing:
    """Sentinel returned by the actor to signal a cache miss."""

    pass


MISSING = _Missing()


@ray.remote
class CacheActor:
    """Named, detached Ray actor serving as a distributed cache backend.

    Internally holds a *cachetools* cache whose values are Ray ``ObjectRef``
    handles.  Workers interact with the cache by passing lightweight
    references rather than full objects — the data itself lives in the Ray
    object store.
    """

    def __init__(self, cache_cls, cache_args=(), cache_kwargs=None):
        if cache_kwargs is None:
            cache_kwargs = {}
        self._cache = cache_cls(*cache_args, **cache_kwargs)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # -- read operations -----------------------------------------------------

    def get(self, key):
        """Return ``ObjRefWrapper`` for *key*, or ``MISSING`` on a miss."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return ObjRefWrapper(self._cache[key])
            self._misses += 1
            return MISSING

    def get_many(self, keys):
        """Batch lookup.  Return ``{key: ObjRefWrapper | MISSING}``."""
        result = {}
        with self._lock:
            for key in keys:
                if key in self._cache:
                    self._hits += 1
                    result[key] = ObjRefWrapper(self._cache[key])
                else:
                    self._misses += 1
                    result[key] = MISSING
        return result

    def contains(self, key):
        with self._lock:
            return key in self._cache

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def len(self):
        with self._lock:
            return len(self._cache)

    def get_maxsize(self):
        return self._cache.maxsize

    def get_currsize(self):
        with self._lock:
            return self._cache.currsize

    def cache_info(self):
        """Return ``(hits, misses, maxsize, currsize)``."""
        with self._lock:
            return (
                self._hits,
                self._misses,
                self._cache.maxsize,
                self._cache.currsize,
            )

    # -- write operations ----------------------------------------------------

    def put(self, key, value):
        """Place *value* in the object store and cache the resulting ref.

        The actor calls :func:`ray.put` so that **it** owns the
        ``ObjectRef``.  This ensures the object survives in the store as
        long as the (detached) actor is alive, regardless of the calling
        worker's lifetime.
        """
        ref = ray.put(value)
        with self._lock:
            self._cache[key] = ref

    def put_many(self, items):
        """Batch insert.  *items* is a list of ``(key, value)`` pairs."""
        refs = [(k, ray.put(v)) for k, v in items]
        with self._lock:
            for k, ref in refs:
                self._cache[k] = ref

    def delete(self, key):
        with self._lock:
            del self._cache[key]

    def delete_many(self, keys):
        """Batch delete.  Silently skips keys that are not present."""
        with self._lock:
            for key in keys:
                self._cache.pop(key, None)

    def pop(self, key):
        """Remove *key* and return its ``ObjRefWrapper``."""
        with self._lock:
            ref = self._cache.pop(key)
            return ObjRefWrapper(ref)

    def popitem(self):
        """Remove and return an arbitrary ``(key, ObjRefWrapper)`` pair."""
        with self._lock:
            key, ref = self._cache.popitem()
            return key, ObjRefWrapper(ref)

    def setdefault(self, key, value):
        """If *key* is absent, store *value* via :func:`ray.put`.  Return the ref wrapper."""
        with self._lock:
            if key in self._cache:
                return ObjRefWrapper(self._cache[key])
        ref = ray.put(value)
        with self._lock:
            if key in self._cache:
                return ObjRefWrapper(self._cache[key])
            self._cache[key] = ref
            return ObjRefWrapper(ref)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
