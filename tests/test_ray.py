"""Ray-specific behaviour tests that have no cachetools equivalent."""

import asyncio
import subprocess
import sys
import textwrap
import uuid

import pytest
import ray

from raycachetools import RayLRUCache

pytestmark = pytest.mark.usefixtures("ray_session")


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@ray.remote(num_cpus=0)
class StartBarrier:
    def __init__(self, parties):
        self._parties = parties
        self._waiting = 0
        self._event = asyncio.Event()

    async def wait(self):
        self._waiting += 1
        if self._waiting >= self._parties:
            self._event.set()
        await self._event.wait()


@ray.remote(num_cpus=0)
def create_cache_after_barrier(name, barrier, key):
    ray.get(barrier.wait.remote())
    cache = RayLRUCache(maxsize=32, name=name)
    cache[key] = key
    return cache.actor_name


# ---------------------------------------------------------------------------
# Actor persistence
# ---------------------------------------------------------------------------


def test_reconnect_to_existing_actor():
    """A second proxy with the same name should see the first proxy's data."""
    name = unique_name("persist")
    cache1 = RayLRUCache(maxsize=10, name=name)
    cache1["hello"] = "world"

    cache2 = RayLRUCache(maxsize=10, name=name)
    assert cache2["hello"] == "world"
    assert len(cache2) == 1


def test_concurrent_first_use_reconnects_to_single_actor():
    """Parallel first users should not race while creating a named actor."""
    name = unique_name("race")
    worker_count = 16
    barrier = StartBarrier.remote(worker_count)

    actor_names = ray.get(
        [
            create_cache_after_barrier.remote(name, barrier, key)
            for key in range(worker_count)
        ],
        timeout=60,
    )

    assert len(set(actor_names)) == 1
    cache = RayLRUCache(maxsize=32, name=name)
    assert cache.get_many(range(worker_count)) == {
        key: key for key in range(worker_count)
    }


def test_data_survives_proxy_deletion():
    """Deleting the proxy object should not kill the actor."""
    name = unique_name("survive")
    cache = RayLRUCache(maxsize=10, name=name)
    cache["key"] = 42
    del cache

    cache2 = RayLRUCache(maxsize=10, name=name)
    assert cache2["key"] == 42


def test_destroy_kills_actor():
    """After destroy(), the actor should be gone."""
    name = unique_name("destroy")
    cache = RayLRUCache(maxsize=10, name=name)
    cache["key"] = 1
    actor_name = cache.actor_name
    cache.destroy()

    # Actor is dead — get_actor should fail
    with pytest.raises(ValueError):
        ray.get_actor(actor_name)


# ---------------------------------------------------------------------------
# Object store
# ---------------------------------------------------------------------------


def test_large_object():
    """Large objects should work since they live in the object store."""
    name = unique_name("large")
    cache = RayLRUCache(maxsize=4, name=name)

    big = list(range(100_000))
    cache["big"] = big
    assert cache["big"] == big


def test_various_types():
    """Different serializable types should round-trip correctly."""
    name = unique_name("types")
    cache = RayLRUCache(maxsize=10, name=name)

    test_values = {
        "int": 42,
        "float": 3.14,
        "str": "hello",
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "tuple": (1, "two", 3.0),
        "none": None,
        "bool": True,
        "bytes": b"binary data",
        "nested": {"a": [1, {"b": 2}]},
    }

    for key, value in test_values.items():
        cache[key] = value

    for key, value in test_values.items():
        assert cache[key] == value, f"Failed for type key={key}"


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


def test_name_property():
    name = unique_name("nameprop")
    cache = RayLRUCache(maxsize=2, name=name)
    assert cache.name == name


def test_actor_name_is_node_local():
    name = unique_name("actorname")
    cache = RayLRUCache(maxsize=2, name=name)

    assert cache.actor_name.endswith(f"_{name}")
    assert cache.hostname


# ---------------------------------------------------------------------------
# Eviction drops refs
# ---------------------------------------------------------------------------


def test_evicted_key_gone():
    """When eviction drops an ObjectRef, the cache should no longer hold it."""
    name = unique_name("evict_ref")
    cache = RayLRUCache(maxsize=2, name=name)

    cache[1] = "a"
    cache[2] = "b"
    cache[3] = "c"  # evicts 1

    assert 1 not in cache
    assert cache[2] == "b"
    assert cache[3] == "c"


# ---------------------------------------------------------------------------
# Object ownership after driver exit
# ---------------------------------------------------------------------------


def test_objects_survive_driver_exit():
    """Cached objects must remain resolvable after the inserting driver exits.

    A separate subprocess populates the cache via a detached actor, then
    exits.  The main process reconnects to the same actor and attempts to
    read back the values.

    If the *driver* (subprocess) owned the ObjectRefs created by
    ``ray.put()``, Ray's reference-counting GC would reclaim the objects
    once that process terminates, leaving the actor with dangling refs.
    """
    name = unique_name("ownership")
    ns = unique_name("ns")
    gcs_address = ray.get_runtime_context().gcs_address

    # A separate driver process populates the cache and then exits.
    # Both processes use the same explicit namespace so they see the
    # same named actor — without this they'd each get their own.
    script = textwrap.dedent(f"""\
        import ray
        ray.init(address="{gcs_address}", namespace="{ns}")

        from raycachetools import RayLRUCache

        cache = RayLRUCache(maxsize=10, name="{name}")
        cache["str_val"] = "hello"
        cache["big_list"] = list(range(500))

        assert len(cache) == 2
        assert cache["str_val"] == "hello"
        print(f"ACTOR_NAME={{cache.actor_name}}")

        # Driver exits — if it owned the ObjectRefs they would be GC'd.
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Subprocess failed (rc={result.returncode}):\n{result.stderr}"
    )
    actor_name = next(
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("ACTOR_NAME=")
    )

    # Reconnect to the actor from the main process.  We use ray.get_actor
    # with an explicit namespace because the main process may be running
    # under a different default namespace.
    from raycachetools._actor import _Missing

    actor = ray.get_actor(actor_name, namespace=ns)

    # The actor should still report 2 entries.
    assert ray.get(actor.len.remote()) == 2

    # The critical check: resolving the ObjectRefs must succeed.
    # If the subprocess owned them, these would fail with ObjectLostError.
    result_str = ray.get(actor.get.remote("str_val"))
    assert not isinstance(result_str, _Missing), "str_val missing from actor"
    assert ray.get(result_str.ref) == "hello"

    result_list = ray.get(actor.get.remote("big_list"))
    assert not isinstance(result_list, _Missing), "big_list missing from actor"
    assert ray.get(result_list.ref) == list(range(500))
