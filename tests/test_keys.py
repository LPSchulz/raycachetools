"""Contract tests for key helpers used by raycachetools decorators."""

import pickle

import pytest

import raycachetools.keys


def test_hashkey_normalizes_kwargs_order():
    key = raycachetools.keys.hashkey
    assert key(1, a=2, b=3) == key(1, b=3, a=2)


def test_methodkey_ignores_instance_argument():
    key = raycachetools.keys.methodkey
    assert key("self-a", 1, x=2) == key("self-b", 1, x=2)


def test_typed_keys_distinguish_numeric_types():
    assert raycachetools.keys.typedkey(1) != raycachetools.keys.typedkey(1.0)
    assert raycachetools.keys.typedmethodkey(
        "self", 1
    ) != raycachetools.keys.typedmethodkey("self", 1.0)


def test_hashkey_rejects_unhashable_arguments():
    with pytest.raises(TypeError):
        hash(raycachetools.keys.hashkey({}))


def test_hashed_tuple_pickle_round_trip_preserves_value_and_hash():
    key = raycachetools.keys.hashkey("abc", q="def")
    original_hash = hash(key)
    cloned = pickle.loads(pickle.dumps(key))
    assert cloned == key
    assert hash(cloned) == original_hash
