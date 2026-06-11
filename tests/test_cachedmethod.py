"""Tests for the @cachedmethod decorator."""

import uuid

import pytest

from raycachetools import RayCache, RayLRUCache, cachedmethod, keys

pytestmark = pytest.mark.usefixtures("ray_session")


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Test helper class
# ---------------------------------------------------------------------------


class Cached:
    def __init__(self, cache, count=0):
        self.cache = cache
        self.count = count

    def __get(self, _value):
        result = self.count
        self.count += 1
        return result

    @cachedmethod(lambda self: self.cache)
    def get(self, value):
        """docstring"""
        return self.__get(value)

    @cachedmethod(lambda self: self.cache, key=keys.typedmethodkey)
    def get_typed(self, value):
        return self.__get(value)

    @cachedmethod(lambda self: self.cache, info=True)
    def get_info(self, value):
        return self.__get(value)


# ---------------------------------------------------------------------------
# Common tests (run for each cache type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator(cache_cls, prefix):
    # Verifies default method key behaviour: equal-hash values (1 and 1.0)
    # share one cache entry.
    cached = Cached(cache_cls(maxsize=2, name=unique_name(prefix)))

    assert cached.get(0) == 0
    assert keys.methodkey(cached, 0) in cached.cache

    assert cached.get(1) == 1
    assert keys.methodkey(cached, 1) in cached.cache

    assert cached.get(1) == 1
    assert cached.get(1.0) == 1

    cached.cache.clear()
    assert cached.get(1) == 2


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator_typed(cache_cls, prefix):
    # Verifies typed key behaviour: 1 and 1.0 are cached independently.
    cached = Cached(cache_cls(maxsize=3, name=unique_name(prefix)))

    assert cached.get_typed(0) == 0
    assert cached.get_typed(1) == 1
    assert cached.get_typed(1) == 1
    assert cached.get_typed(1.0) == 2
    assert keys.typedmethodkey(cached, 1) in cached.cache
    assert keys.typedmethodkey(cached, 1.0) in cached.cache
    assert cached.get_typed(1.0) == 2
    assert cached.get_typed(0.0) == 3


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator_info(cache_cls, prefix):
    # Verifies per-wrapper hit/miss accounting and cache_clear reset behaviour.
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    cached = Cached(cache)
    wrapper = cached.get_info

    info = wrapper.cache_info()
    assert info.hits == 0
    assert info.misses == 0

    assert wrapper(0) == 0
    info = wrapper.cache_info()
    assert info.misses == 1

    assert wrapper(1) == 1
    info = wrapper.cache_info()
    assert info.misses == 2

    assert wrapper(0) == 0
    info = wrapper.cache_info()
    assert info.hits == 1
    assert info.misses == 2

    wrapper.cache_clear()
    assert len(cache) == 0
    info = wrapper.cache_info()
    assert info.hits == 0
    assert info.misses == 0

    assert wrapper(0) == 2
    info = wrapper.cache_info()
    assert info.misses == 1


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator_wrapped(cache_cls, prefix):
    # Verifies functools metadata forwarding and that __wrapped__ bypasses cache.
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    cached = Cached(cache)

    assert len(cache) == 0
    assert cached.get.__wrapped__(0) == 0
    assert cached.get.__wrapped__.__self__ is cached
    assert cached.get.__name__ == "get"
    assert cached.get.__doc__.strip() == "docstring"
    assert len(cache) == 0
    assert cached.get(0) == 1
    assert len(cache) == 1
    assert cached.get(0) == 1
    assert len(cache) == 1


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator_attributes(cache_cls, prefix):
    # Verifies wrapper exposes the effective cache object and key function.
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    cached = Cached(cache)

    assert cached.get.cache is cache
    assert cached.get.cache_key is keys.methodkey


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_decorator_clear(cache_cls, prefix):
    # Verifies cache_clear flushes cache state.
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    cached = Cached(cache)

    assert cached.get(0) == 0
    assert len(cache) == 1
    cached.get.cache_clear()
    assert len(cache) == 0


def test_decorator_different_names():
    with pytest.raises(Exception):

        class Bad:
            @cachedmethod(lambda _: None)
            def foo(_self):
                pass

            bar = foo


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cm_cache", id="RayCache"),
        pytest.param(RayLRUCache, "cm_lru", id="RayLRUCache"),
    ],
)
def test_shared_cache(cache_cls, prefix):
    # Verifies wrappers keep independent counters even when sharing one cache.
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    cached1 = Cached(cache)
    cached2 = Cached(cache)

    info1 = cached1.get_info
    info2 = cached2.get_info

    assert info1.cache_info().hits == 0
    assert info2.cache_info().hits == 0

    assert info1(0) == 0
    assert info1.cache_info().misses == 1
    assert info2.cache_info().misses == 0

    # cached2 should hit since cache is shared
    assert info2(0) == 0
    assert info2.cache_info().hits == 1


# ---------------------------------------------------------------------------
# None-cache tests
# ---------------------------------------------------------------------------


def test_none_info():
    cached = Cached(None)
    wrapper = cached.get_info
    with pytest.raises(TypeError, match="cache\(self\) must return a mutable mapping"):
        wrapper.cache_info()


def test_decorator_rejects_lock_and_condition_arguments():
    with pytest.raises(TypeError):
        cachedmethod(lambda self: self.cache, lock=None)
    with pytest.raises(TypeError):
        cachedmethod(lambda self: self.cache, condition=None)
