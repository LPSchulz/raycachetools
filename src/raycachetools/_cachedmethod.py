"""Method decorator helpers for raycachetools."""

__all__ = ()

import functools


class _Wrapper:
    """Bound wrapper around a cached method invocation."""

    def __init__(self, method, cache, key, info=None):
        functools.update_wrapper(self, method)
        self._method = method
        self._cache = cache
        self._key = key
        self._info = info
        self._hits = 0
        self._misses = 0

    def __call__(self, *args, **kwargs):
        c = self._cache
        if c is None:
            self._misses += 1
            return self._method(*args, **kwargs)
        k = self._key(self._method.__self__, *args, **kwargs)
        try:
            result = c[k]
            self._hits += 1
            return result
        except KeyError:
            self._misses += 1
        v = self._method(*args, **kwargs)
        try:
            c[k] = v
        except ValueError:
            pass  # value too large
        return v

    def cache_clear(self):
        c = self._cache
        if c is not None:
            c.clear()
        self._hits = 0
        self._misses = 0

    @property
    def cache(self):
        return self._cache

    @property
    def cache_key(self):
        return self._key

    def cache_info(self):
        if self._info is None:
            return None
        return self._info(self._cache, self._hits, self._misses)


class _Descriptor:
    """Descriptor that creates per-instance :class:`_Wrapper` objects."""

    def __init__(self, method, cache_factory, key, info=None):
        self._method = method
        self._cache_factory = cache_factory
        self._key = key
        self._info = info
        self._attrname = None
        functools.update_wrapper(self, method)

    def __set_name__(self, owner, name):
        if self._attrname is None:
            self._attrname = name
        elif name != self._attrname:
            raise TypeError(
                "Cannot assign the same @cachedmethod to two different names "
                f"({self._attrname!r} and {name!r})."
            )

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cache = self._cache_factory(obj)
        bound = self._method.__get__(obj, objtype)
        wrapper = _Wrapper(bound, cache, self._key, self._info)
        # Cache the wrapper on the instance for future access
        if self._attrname is not None:
            try:
                obj.__dict__[self._attrname] = wrapper
            except (AttributeError, TypeError):
                pass
        return wrapper


def _wrapper(method, cache, key, info=None):
    return _Descriptor(method, cache, key, info)
