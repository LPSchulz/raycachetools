"""Function decorator helpers for raycachetools.

No local locking is needed — the Ray actor serialises all cache access.
Hit / miss counters are tracked *locally* (per-worker) when ``info=True``.
"""

__all__ = ()

import functools


def _info(func, cache, key, info):
    """Wrapper that tracks hits / misses and exposes ``cache_info``."""
    hits = misses = 0

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        k = key(*args, **kwargs)
        try:
            result = cache[k]
            hits += 1
            return result
        except KeyError:
            misses += 1
        v = func(*args, **kwargs)
        try:
            cache[k] = v
        except ValueError:
            pass  # value too large
        return v

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = misses = 0

    def cache_info():
        return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _plain(func, cache, key):
    """Simple wrapper — no hit / miss tracking."""

    def wrapper(*args, **kwargs):
        k = key(*args, **kwargs)
        try:
            return cache[k]
        except KeyError:
            pass
        v = func(*args, **kwargs)
        try:
            cache[k] = v
        except ValueError:
            pass  # value too large
        return v

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def _uncached_info(func, info):
    misses = 0

    def wrapper(*args, **kwargs):
        nonlocal misses
        misses += 1
        return func(*args, **kwargs)

    def cache_clear():
        nonlocal misses
        misses = 0

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = lambda: info(0, misses)
    return wrapper


def _uncached(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.cache_clear = lambda: None
    return wrapper


def _wrapper(func, cache, key, info=None):
    if info is not None:
        if cache is None:
            wrapper = _uncached_info(func, info)
        else:
            wrapper = _info(func, cache, key, info)
    else:
        if cache is None:
            wrapper = _uncached(func)
        else:
            wrapper = _plain(func, cache, key)
        wrapper.cache_info = None

    wrapper.cache = cache
    wrapper.cache_key = key

    return functools.update_wrapper(wrapper, func)
