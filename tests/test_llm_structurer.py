from cerebro.cache import Cache
from cerebro.llm.providers import MockProvider
from cerebro.structure.llm import LLMStructurer, chunk_transcript
from cerebro.transcript import Segment, Transcript


def _transcript():
    segs = [
        Segment(text=" ".join(["word"] * 300), start=0.0, duration=60.0),
        Segment(text=" ".join(["idea"] * 300), start=60.0, duration=60.0),
    ]
    return Transcript(source="x", title="Test Video", segments=segs)


def test_chunking_respects_word_budget():
    chunks = chunk_transcript(_transcript(), max_words=250)
    assert len(chunks) == 2  # 300-word segments each exceed the 250 budget


def test_llm_structurer_builds_hierarchy():
    mm = LLMStructurer(MockProvider()).structure(_transcript(), level="full")
    assert mm.node_count() > 1
    assert mm.depth() >= 2
    assert mm.root.title


def test_expert_adds_relationships():
    mm = LLMStructurer(MockProvider()).structure(_transcript(), level="expert")
    assert len(mm.relationships) >= 1


def test_cache_avoids_recompute(tmp_path):
    cache = Cache(root=tmp_path / "c", enabled=True)
    provider = MockProvider()
    structurer = LLMStructurer(provider, cache=cache)

    structurer.structure(_transcript(), level="full")
    calls_after_first = provider.calls
    assert calls_after_first > 0

    structurer.structure(_transcript(), level="full")
    assert provider.calls == calls_after_first  # fully served from cache
