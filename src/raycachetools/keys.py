"""Key functions for memoizing decorators."""

__all__ = ("hashkey", "methodkey", "typedkey", "typedmethodkey")


class _HashedTuple(tuple):
    """A tuple subclass that caches its hash value.

    Cache decorators may hash the key multiple times on a miss; caching the
    hash avoids redundant work.  Identical to the cachetools implementation.
    """

    __hashvalue = None

    def __hash__(self, hash=tuple.__hash__):
        hashvalue = self.__hashvalue
        if hashvalue is None:
            self.__hashvalue = hashvalue = hash(self)
        return hashvalue

    def __add__(self, other, add=tuple.__add__):
        return _HashedTuple(add(self, other))

    def __radd__(self, other, add=tuple.__add__):
        return _HashedTuple(add(other, self))

    def __getstate__(self):
        return {}


_kwmark = (_HashedTuple,)


def hashkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments."""
    if kwargs:
        return _HashedTuple(args + _kwmark + tuple(sorted(kwargs.items())))
    else:
        return _HashedTuple(args)


def methodkey(self, *args, **kwargs):
    """Return a cache key for use with cached methods."""
    return hashkey(*args, **kwargs)


def typedkey(*args, **kwargs):
    """Return a typed cache key for the specified hashable arguments."""
    if kwargs:
        sorted_kwargs = tuple(sorted(kwargs.items()))
        key = _HashedTuple(args + _kwmark + sorted_kwargs)
        key += tuple(type(v) for _, v in sorted_kwargs)
    else:
        key = _HashedTuple(args)
    key += tuple(type(v) for v in args)
    return key


def typedmethodkey(self, *args, **kwargs):
    """Return a typed cache key for use with cached methods."""
    return typedkey(*args, **kwargs)
