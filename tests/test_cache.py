from cerebro.cache import Cache
from cerebro.paths import CACHE_DIR


def test_default_root_is_stable_global_path_not_cwd():
    cache = Cache(enabled=False)  # enabled=False avoids creating the real dir
    assert cache.root == CACHE_DIR
    assert cache.root.is_absolute()


def test_stats_and_clear_roundtrip(tmp_path):
    cache = Cache(root=tmp_path / "cache")
    assert cache.stats() == (0, 0)

    cache.set(Cache.key("a"), {"x": 1})
    cache.set(Cache.key("b"), {"y": 2})
    count, total_bytes = cache.stats()
    assert count == 2
    assert total_bytes > 0

    removed = cache.clear()
    assert removed == 2
    assert cache.stats() == (0, 0)


def test_clear_on_empty_or_missing_dir_is_a_noop(tmp_path):
    cache = Cache(root=tmp_path / "never_created", enabled=False)
    assert cache.clear() == 0
    assert cache.stats() == (0, 0)
