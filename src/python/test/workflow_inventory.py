"""Retain pytest's collection before marker deselection for the Q inventory audit."""
import pytest

COLLECTED = pytest.StashKey[frozenset[str]]()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    config.stash[COLLECTED] = frozenset(item.nodeid.split("[", 1)[0] for item in items)
