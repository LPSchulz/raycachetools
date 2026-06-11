# raycachetools

Ray-distributed memoizing collections and decorators, built on top of
[cachetools](https://github.com/tkem/cachetools).

`raycachetools` extends `cachetools` by backing every cache with a named,
detached [Ray](https://www.ray.io/) actor.  Each cache is **local to one Ray
node**: workers on the same machine share the same LRU, TTL, FIFO, LFU, or RR
cache, while other nodes keep their own local copy.  This keeps ownership and
object-store traffic local and avoids funneling all cache traffic through a
single node.

```python
import time
import ray
from raycachetools import RayLRUCache, cached

ray.init(namespace="my-app")

CACHE_NAME = "example-shared-lru"

@ray.remote
def worker(x):
    cache = RayLRUCache(maxsize=128, name=CACHE_NAME)

    @cached(cache=cache)
    def expensive_square(n):
        time.sleep(0.2)  # stand-in for expensive work
        return n * n

    return expensive_square(x)

# Repeated inputs can be served from the node-local cache
results = ray.get([worker.remote(i) for i in [1, 2, 3, 1, 2, 3]])
print(results)  # [1, 4, 9, 1, 4, 9]
```

The important usage rule is: **create the cache handle inside the remote
function or Ray actor method that uses it**.  Do not create a cache handle on
the main driver and pass that handle into remote functions, because that binds
those remote workers to the driver's node-local cache actor.

## Installation

raycachetools is available from PyPI and can be installed by running:

```
pip install raycachetools
```

It requires Python &ge; 3.10, [Ray](https://pypi.org/project/ray/) &ge; 2.0,
and [cachetools](https://pypi.org/project/cachetools/) &ge; 7.0.

## Important: Ray namespaces

Named actors in Ray are scoped to a **namespace**.  If you don't pass an
explicit `namespace` to `ray.init()`, each process gets its own anonymous
namespace — which means each process creates a *separate* per-node actor even
when using the same cache `name`.  To reconnect to the same local cache on the
same node across multiple drivers or jobs, **always specify a namespace**:

```python
ray.init(address="auto", namespace="my-app")
```

All processes that should share the same caches must use the same namespace.

## Key differences from cachetools

| Feature | cachetools | raycachetools |
|---|---|---|
| Process scope | In-process only | Shared by workers on the same Ray node |
| `name` parameter | N/A | Every cache requires a unique `name` |
| Thread safety | User provides `lock` / `condition` | Actor serialises cache access; no cache lock needed |
| `getsizeof` support | Supported | Not supported (size = number of items) |
| Concurrent misses | `condition` can prevent duplicate work | Duplicate work is possible; actor serialises cache access only |
| Cache lifetime | Follows Python object lifetime | Detached per-node actor survives the creator; call `destroy()` to clean up |

## Documentation

Full documentation is available at
[raycachetools.readthedocs.io](https://raycachetools.readthedocs.io/).

If you are not yet familiar with cache algorithms and the decorator API, start
with the [cachetools documentation](https://cachetools.readthedocs.io/) — the
raycachetools docs focus on the differences and additions only.

## Related projects

- [cachetools](https://pypi.org/project/cachetools/) — the upstream library
  this project extends.
- [Ray](https://pypi.org/project/ray/) — the distributed computing framework
  used for the actor backend.

## License

MIT
