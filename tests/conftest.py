import pytest
import ray


@pytest.fixture(scope="session")
def ray_session():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
