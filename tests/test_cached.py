"""Tests for the @cached decorator."""

import uuid

import pytest

import raycachetools
import raycachetools.keys
from raycachetools import RayCache, RayLRUCache

pytestmark = pytest.mark.usefixtures("ray_session")


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Shared count helper
# ---------------------------------------------------------------------------


def _make_counter():
    """Return a callable that increments a counter on each call."""
    state = {"count": -1}

    def func(*args, **kwargs):
        state["count"] += 1
        return state["count"]

    return func


# ---------------------------------------------------------------------------
# Common decorator tests (run for each cache type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cached_c", id="RayCache"),
        pytest.param(RayLRUCache, "cached_lru", id="RayLRUCache"),
    ],
)
def test_decorator(cache_cls, prefix):
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    func = _make_counter()
    wrapper = raycachetools.cached(cache)(func)

    assert len(cache) == 0
    assert wrapper(0) == 0
    assert len(cache) == 1
    assert raycachetools.keys.hashkey(0) in cache
    assert raycachetools.keys.hashkey(1) not in cache

    assert wrapper(1) == 1
    assert len(cache) == 2
    assert raycachetools.keys.hashkey(0) in cache
    assert raycachetools.keys.hashkey(1) in cache
    # 1 and 1.0 hash equally
    assert raycachetools.keys.hashkey(1.0) in cache

    assert wrapper(1) == 1
    assert len(cache) == 2

    assert wrapper(1.0) == 1
    assert len(cache) == 2


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cached_c", id="RayCache"),
        pytest.param(RayLRUCache, "cached_lru", id="RayLRUCache"),
    ],
)
def test_decorator_typed(cache_cls, prefix):
    cache = cache_cls(maxsize=3, name=unique_name(prefix))
    func = _make_counter()
    key = raycachetools.keys.typedkey
    wrapper = raycachetools.cached(cache, key=key)(func)

    assert len(cache) == 0
    assert wrapper(0) == 0
    assert len(cache) == 1

    assert wrapper(1) == 1
    assert len(cache) == 2

    assert wrapper(1) == 1
    assert len(cache) == 2

    assert wrapper(1.0) == 2
    assert len(cache) == 3

    assert wrapper(1.0) == 2
    assert len(cache) == 3


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cached_c", id="RayCache"),
        pytest.param(RayLRUCache, "cached_lru", id="RayLRUCache"),
    ],
)
def test_decorator_wrapped(cache_cls, prefix):
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    func = _make_counter()
    wrapper = raycachetools.cached(cache)(func)

    assert wrapper.__wrapped__ is func

    assert len(cache) == 0
    assert wrapper.__wrapped__(0) == 0
    assert len(cache) == 0
    assert wrapper(0) == 1
    assert len(cache) == 1
    assert wrapper(0) == 1
    assert len(cache) == 1


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cached_c", id="RayCache"),
        pytest.param(RayLRUCache, "cached_lru", id="RayLRUCache"),
    ],
)
def test_decorator_attributes(cache_cls, prefix):
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    func = _make_counter()
    wrapper = raycachetools.cached(cache)(func)

    assert wrapper.cache is cache
    assert wrapper.cache_key is raycachetools.keys.hashkey


@pytest.mark.parametrize(
    "cache_cls,prefix",
    [
        pytest.param(RayCache, "cached_c", id="RayCache"),
        pytest.param(RayLRUCache, "cached_lru", id="RayLRUCache"),
    ],
)
def test_decorator_clear(cache_cls, prefix):
    cache = cache_cls(maxsize=2, name=unique_name(prefix))
    func = _make_counter()
    wrapper = raycachetools.cached(cache)(func)
    assert wrapper(0) == 0
    assert len(cache) == 1
    wrapper.cache_clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Cache-info tests (not parametrized)
# ---------------------------------------------------------------------------


def test_cache_info():
    cache = RayCache(maxsize=2, name=unique_name("cached_info"))
    func = _make_counter()
    wrapper = raycachetools.cached(cache, info=True)(func)
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


def test_lru_cache_info():
    cache = RayLRUCache(maxsize=2, name=unique_name("cached_lru_info"))
    func = _make_counter()
    wrapper = raycachetools.cached(cache, info=True)(func)
    info = wrapper.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.maxsize == 2
    assert info.currsize == 0

    assert wrapper(0) == 0
    info = wrapper.cache_info()
    assert info.misses == 1
    assert info.currsize == 1


# ---------------------------------------------------------------------------
# None-cache tests
# ---------------------------------------------------------------------------


def test_none_decorator():
    func = lambda *args, **kwargs: args + tuple(kwargs.items())
    wrapper = raycachetools.cached(None)(func)

    assert wrapper(0) == (0,)
    assert wrapper(1) == (1,)
    assert wrapper(1, foo="bar") == (1, ("foo", "bar"))


def test_none_decorator_attributes():
    func = lambda *args, **kwargs: args + tuple(kwargs.items())
    wrapper = raycachetools.cached(None)(func)

    assert wrapper.cache is None
    assert wrapper.cache_key is raycachetools.keys.hashkey


def test_none_decorator_clear():
    func = lambda *args, **kwargs: args + tuple(kwargs.items())
    wrapper = raycachetools.cached(None)(func)
    wrapper.cache_clear()  # no-op


def test_decorator_rejects_lock_and_condition_arguments():
    cache = RayCache(maxsize=2, name=unique_name("cached_args"))

    with pytest.raises(TypeError):
        raycachetools.cached(cache, lock=None)
    with pytest.raises(TypeError):
        raycachetools.cached(cache, condition=None)
