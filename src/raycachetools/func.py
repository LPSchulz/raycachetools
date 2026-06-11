"""`functools.lru_cache` compatible memoizing function decorators.

Each decorator creates (or connects to) a named Ray actor backed by the
corresponding cachetools eviction policy.
"""

__all__ = ("fifo_cache", "lfu_cache", "lru_cache", "rr_cache", "ttl_cache")

import random
import time

from . import (
    RayFIFOCache,
    RayLFUCache,
    RayLRUCache,
    RayRRCache,
    RayTTLCache,
    cached,
    keys,
)


def _make_name(func, prefix):
    """Derive a deterministic actor name from the function's qualified name."""
    return f"_raycache_{prefix}_{func.__module__}.{func.__qualname__}"


def _cache(cache, maxsize, typed):
    def decorator(func):
        key = keys.typedkey if typed else keys.hashkey
        wrapper = cached(cache=cache, key=key, info=True)(func)
        wrapper.cache_parameters = lambda: {"maxsize": maxsize, "typed": typed}
        return wrapper

    return decorator


def fifo_cache(maxsize=128, typed=False, name=None):
    """Decorator: memoize results using a distributed FIFO cache.

    Parameters
    ----------
    maxsize : int or None
        Maximum number of entries.  ``None`` means unlimited (plain dict
        on the actor).
    typed : bool
        If *True*, arguments of different types are cached separately.
    name : str or None
        Actor name.  Derived from the function's qualified name when *None*.
    """
    if callable(maxsize):
        # Called as bare ``@fifo_cache`` without parentheses.
        func, maxsize = maxsize, 128
        n = name or _make_name(func, "fifo")
        return _cache(RayFIFOCache(128, name=n), 128, typed)(func)

    def decorator(func):
        n = name or _make_name(func, "fifo")
        if maxsize is None:
            return _cache(RayFIFOCache(2**31, name=n), None, typed)(func)
        return _cache(RayFIFOCache(maxsize, name=n), maxsize, typed)(func)

    return decorator


def lfu_cache(maxsize=128, typed=False, name=None):
    """Decorator: memoize results using a distributed LFU cache."""
    if callable(maxsize):
        func, maxsize = maxsize, 128
        n = name or _make_name(func, "lfu")
        return _cache(RayLFUCache(128, name=n), 128, typed)(func)

    def decorator(func):
        n = name or _make_name(func, "lfu")
        if maxsize is None:
            return _cache(RayLFUCache(2**31, name=n), None, typed)(func)
        return _cache(RayLFUCache(maxsize, name=n), maxsize, typed)(func)

    return decorator


def lru_cache(maxsize=128, typed=False, name=None):
    """Decorator: memoize results using a distributed LRU cache."""
    if callable(maxsize):
        func, maxsize = maxsize, 128
        n = name or _make_name(func, "lru")
        return _cache(RayLRUCache(128, name=n), 128, typed)(func)

    def decorator(func):
        n = name or _make_name(func, "lru")
        if maxsize is None:
            return _cache(RayLRUCache(2**31, name=n), None, typed)(func)
        return _cache(RayLRUCache(maxsize, name=n), maxsize, typed)(func)

    return decorator


def rr_cache(maxsize=128, choice=random.choice, typed=False, name=None):
    """Decorator: memoize results using a distributed Random Replacement cache."""
    if callable(maxsize):
        func, maxsize = maxsize, 128
        n = name or _make_name(func, "rr")
        return _cache(RayRRCache(128, name=n, choice=choice), 128, typed)(func)

    def decorator(func):
        n = name or _make_name(func, "rr")
        if maxsize is None:
            return _cache(RayRRCache(2**31, name=n, choice=choice), None, typed)(func)
        return _cache(RayRRCache(maxsize, name=n, choice=choice), maxsize, typed)(func)

    return decorator


def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False, name=None):
    """Decorator: memoize results using a distributed TTL + LRU cache."""
    if callable(maxsize):
        func, maxsize = maxsize, 128
        n = name or _make_name(func, "ttl")
        return _cache(RayTTLCache(128, ttl, name=n, timer=timer), 128, typed)(func)

    def decorator(func):
        n = name or _make_name(func, "ttl")
        if maxsize is None:
            return _cache(RayTTLCache(2**31, ttl, name=n, timer=timer), None, typed)(
                func
            )
        return _cache(RayTTLCache(maxsize, ttl, name=n, timer=timer), maxsize, typed)(
            func
        )

    return decorator
