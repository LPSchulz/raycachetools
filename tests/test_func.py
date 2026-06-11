"""Tests for func.py convenience decorators (lru_cache, etc.)."""

import uuid

import pytest

import raycachetools.func

pytestmark = pytest.mark.usefixtures("ray_session")


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _decorate(decorator, maxsize, **kwargs):
    """Helper that injects a unique actor name."""
    kwargs.setdefault("name", unique_name("func"))
    return decorator(maxsize, **kwargs)


# ---------------------------------------------------------------------------
# Common tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorator",
    [
        pytest.param(raycachetools.func.fifo_cache, id="fifo_cache"),
        pytest.param(raycachetools.func.lfu_cache, id="lfu_cache"),
        pytest.param(raycachetools.func.lru_cache, id="lru_cache"),
        pytest.param(raycachetools.func.rr_cache, id="rr_cache"),
        pytest.param(raycachetools.func.ttl_cache, id="ttl_cache"),
    ],
)
def test_decorator(decorator):
    cached = _decorate(decorator, maxsize=2)(lambda n: n)
    assert cached.cache_parameters() == {"maxsize": 2, "typed": False}
    info = cached.cache_info()
    assert info.hits == 0
    assert info.misses == 0

    assert cached(1) == 1
    info = cached.cache_info()
    assert info.misses == 1

    assert cached(1) == 1
    info = cached.cache_info()
    assert info.hits == 1

    # 1.0 == 1 so it should be a hit
    assert cached(1.0) == 1
    info = cached.cache_info()
    assert info.hits == 2


@pytest.mark.parametrize(
    "decorator",
    [
        pytest.param(raycachetools.func.fifo_cache, id="fifo_cache"),
        pytest.param(raycachetools.func.lfu_cache, id="lfu_cache"),
        pytest.param(raycachetools.func.lru_cache, id="lru_cache"),
        pytest.param(raycachetools.func.rr_cache, id="rr_cache"),
        pytest.param(raycachetools.func.ttl_cache, id="ttl_cache"),
    ],
)
def test_decorator_clear(decorator):
    cached = _decorate(decorator, maxsize=2)(lambda n: n)
    assert cached(1) == 1
    info = cached.cache_info()
    assert info.misses == 1

    cached.cache_clear()
    info = cached.cache_info()
    assert info.hits == 0
    assert info.misses == 0

    assert cached(1) == 1
    info = cached.cache_info()
    assert info.misses == 1


@pytest.mark.parametrize(
    "decorator",
    [
        pytest.param(raycachetools.func.fifo_cache, id="fifo_cache"),
        pytest.param(raycachetools.func.lfu_cache, id="lfu_cache"),
        pytest.param(raycachetools.func.lru_cache, id="lru_cache"),
        pytest.param(raycachetools.func.rr_cache, id="rr_cache"),
        pytest.param(raycachetools.func.ttl_cache, id="ttl_cache"),
    ],
)
def test_decorator_typed(decorator):
    cached = _decorate(decorator, maxsize=2, typed=True)(lambda n: n)
    assert cached.cache_parameters() == {"maxsize": 2, "typed": True}
    assert cached(1) == 1
    info = cached.cache_info()
    assert info.misses == 1

    assert cached(1) == 1
    info = cached.cache_info()
    assert info.hits == 1

    # float 1.0 is a different type → another miss
    assert cached(1.0) == 1.0
    info = cached.cache_info()
    assert info.misses == 2

    assert cached(1.0) == 1.0
    info = cached.cache_info()
    assert info.hits == 2


def test_default_name_reconnects_same_function_cache():
    def square(n):
        return n * n

    wrapped1 = raycachetools.func.lru_cache(maxsize=8)(square)
    wrapped2 = raycachetools.func.lru_cache(maxsize=8)(square)

    wrapped1.cache.clear()
    assert wrapped1(7) == 49
    assert wrapped2(7) == 49
    assert wrapped2.cache_info().hits == 1
