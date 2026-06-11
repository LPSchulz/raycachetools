"""Tests for raycachetools proxy behavior (not cachetools policy logic)."""

import uuid

import pytest
import ray

import raycachetools as rct
from raycachetools import (
    RayCache,
    RayFIFOCache,
    RayLFUCache,
    RayLRUCache,
    RayRRCache,
    RayTLRUCache,
    RayTTLCache,
)

pytestmark = pytest.mark.usefixtures("ray_session")


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


CACHE_CASES = [
    pytest.param(RayCache, {}, "cache", id="RayCache"),
    pytest.param(RayFIFOCache, {}, "fifo", id="RayFIFOCache"),
    pytest.param(RayLFUCache, {}, "lfu", id="RayLFUCache"),
    pytest.param(RayLRUCache, {}, "lru", id="RayLRUCache"),
    pytest.param(RayRRCache, {}, "rr", id="RayRRCache"),
    pytest.param(RayTTLCache, {"ttl": 3600}, "ttl", id="RayTTLCache"),
    pytest.param(
        RayTLRUCache,
        {"ttu": lambda _key, _value, now: now + 3600},
        "tlru",
        id="RayTLRUCache",
    ),
]


@pytest.mark.parametrize("cache_cls,options,prefix", CACHE_CASES)
def test_cache_proxy_exposes_actor_backed_basics(cache_cls, options, prefix):
    kwargs = dict(options)
    kwargs["name"] = unique_name(prefix)
    cache = cache_cls(1, **kwargs)
    node_id = ray.get_runtime_context().get_node_id()
    node = next(node for node in ray.nodes() if node["NodeID"] == node_id)
    hostname = node["NodeManagerHostname"]

    assert len(cache) == 0
    assert cache.maxsize == 1
    assert cache.currsize == 0
    assert cache.name == kwargs["name"]
    assert cache.hostname == hostname
    assert cache.actor_name == f"{rct._sanitize_hostname(hostname)}_{kwargs['name']}"
    assert repr(cache).startswith(cache.__class__.__name__)


def test_mapping_operations_round_trip_through_actor():
    cache = RayCache(maxsize=3, name=unique_name("mapping"))

    cache["a"] = 1
    cache["b"] = 2
    assert len(cache) == 2
    assert cache["a"] == 1
    assert cache.get("missing") is None
    assert cache.get("missing", 99) == 99

    assert cache.setdefault("a", 42) == 1
    assert cache.setdefault("c", 3) == 3
    assert set(cache) == {"a", "b", "c"}

    assert cache.pop("b") == 2
    assert "b" not in cache

    key, value = cache.popitem()
    assert key in {"a", "c"}
    assert value in {1, 3}

    remaining = "c" if key == "a" else "a"
    del cache[remaining]
    assert remaining not in cache


def test_missing_key_errors_are_translated_consistently():
    cache = RayCache(maxsize=2, name=unique_name("keyerrors"))

    with pytest.raises(KeyError):
        _ = cache["missing"]
    with pytest.raises(KeyError):
        del cache["missing"]
    with pytest.raises(KeyError):
        cache.pop("missing")

    assert cache.pop("missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def test_get_many():
    cache = RayCache(maxsize=10, name=unique_name("getmany"))
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    result = cache.get_many(["a", "c", "missing"])
    assert result == {"a": 1, "c": 3}


def test_get_many_empty():
    cache = RayCache(maxsize=10, name=unique_name("getmany_empty"))
    assert cache.get_many(["x", "y"]) == {}


def test_put_many():
    cache = RayCache(maxsize=10, name=unique_name("putmany"))
    cache.put_many([("x", 10), ("y", 20), ("z", 30)])

    assert len(cache) == 3
    assert cache["x"] == 10
    assert cache["y"] == 20
    assert cache["z"] == 30


def test_delete_many():
    cache = RayCache(maxsize=10, name=unique_name("delmany"))
    cache.put_many([("a", 1), ("b", 2), ("c", 3)])

    cache.delete_many(["a", "c", "nonexistent"])
    assert len(cache) == 1
    assert cache["b"] == 2


def test_batch_round_trip():
    """put_many then get_many in a single pair of RPCs."""
    cache = RayLRUCache(maxsize=100, name=unique_name("batch_rt"))
    items = [(str(i), i * i) for i in range(50)]
    cache.put_many(items)

    keys = [str(i) for i in range(50)]
    result = cache.get_many(keys)
    assert len(result) == 50
    for k, v in items:
        assert result[k] == v
