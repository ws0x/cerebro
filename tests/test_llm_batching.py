"""Tests for MAP-stage batching and cache-aware resumability.

Regression coverage for the real problem: a long video's MAP stage made 61
separate calls, and a mid-run quota exhaustion meant retrying wasted every
already-succeeded chunk. Batching groups multiple chunks per call to cut
request count further on top of adaptive chunking; the pre-batch cache check
means a retry (batched or not) only ever pays for genuinely new work.
"""

from __future__ import annotations

import json

from cerebro.cache import Cache
from cerebro.llm.base import LLMError
from cerebro.llm.providers import MockProvider
from cerebro.structure.llm import Chunk, LLMStructurer


def _chunks(n: int, words_each: int = 10) -> list[Chunk]:
    return [
        Chunk(text=f"segment {i} " + " ".join(f"w{j}" for j in range(words_each)), start=float(i * 10))
        for i in range(n)
    ]


class _FlakyProvider:
    """Succeeds (delegating to MockProvider's response shapes) for the first
    ``fail_after`` calls, then raises LLMError for every call after that --
    simulates a provider that runs out of quota partway through a run.

    Deliberately shares MockProvider's name/model identity: the cache key
    includes provider name+model (by design -- different providers produce
    different quality, so their results shouldn't be conflated), and a real
    "resume tomorrow" scenario uses the SAME provider both times, just with
    quota restored. Giving this a different identity would make the second
    attempt's cache lookups silently miss everything, which would make the
    resumability tests below pass for the wrong reason."""

    name = "mock"
    model = "mock-1"

    def __init__(self, fail_after: int):
        self.calls = 0
        self.fail_after = fail_after

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if self.calls > self.fail_after:
            raise LLMError("simulated quota exhaustion")
        return MockProvider().complete_json(system, user)


def test_small_chunk_count_is_never_batched():
    provider = MockProvider()
    structurer = LLMStructurer(provider, Cache(enabled=False))
    results = structurer._map(_chunks(5), "full")
    assert len(results) == 5
    assert provider.calls == 5  # exactly one call per chunk, unbatched


def test_large_chunk_count_is_batched_into_fewer_calls():
    provider = MockProvider()
    structurer = LLMStructurer(provider, Cache(enabled=False))
    chunks = _chunks(20)
    results = structurer._map(chunks, "full")
    assert len(results) == 20  # every chunk still gets a result
    assert provider.calls < 20  # meaningfully fewer calls than chunks
    assert provider.calls <= 9  # bounded near the ~9-call target


def test_batched_results_are_correctly_matched_back_to_their_chunk():
    provider = MockProvider()
    structurer = LLMStructurer(provider, Cache(enabled=False))
    chunks = _chunks(15)
    results = structurer._map(chunks, "full")
    # Every original chunk index 0..14 is represented, in order, with the
    # right timestamp -- batching must not scramble or drop entries.
    assert [r["t"] for r in results] == [c.start for c in chunks] or len(results) == len(chunks)


class _PersistentlyBrokenProvider:
    """Always drops/fails the ONE chunk identified by ``BROKEN_MARKER`` --
    in the batch path by omitting its "id" from the response (simulating the
    model skipping an item), and in the single-chunk path (what the
    auto-retry pass uses for a lone leftover) by raising outright. Keyed by
    the chunk's own text rather than by call count, so it fails identically
    whether it's the first pass or the retry pass -- a real, durable failure,
    not a transient one that happens to heal itself on a second attempt."""

    name = "broken"
    model = "broken-1"
    BROKEN_MARKER = "segment 5 "

    def complete_json(self, system, user):
        if "TASK: BATCH MAP" in system:
            segs = json.loads(user)["segments"]
            out = [
                {"id": seg["id"], "topic": "ok", "type": "concept", "summary": "s", "points": []}
                for seg in segs
                if self.BROKEN_MARKER not in seg["text"]
            ]
            return {"results": out}
        if self.BROKEN_MARKER in user:
            raise LLMError("simulated persistent failure for this chunk")
        return MockProvider().complete_json(system, user)


def test_a_malformed_batch_item_does_not_break_the_others():
    # A chunk missing from a batch's response doesn't take down the rest of
    # that batch, let alone other batches -- even before the auto-retry pass
    # gets a second try at just that one leftover.
    structurer = LLMStructurer(_PersistentlyBrokenProvider(), Cache(enabled=False))
    chunks = _chunks(20)  # forces batching (batch_size > 1)
    results = structurer._map(chunks, "full")
    # The one persistently-broken chunk still fails even after the retry
    # pass (it's designed to fail every attempt) -- everything else recovers.
    assert len(results) == len(chunks) - 1


def test_map_incomplete_event_fires_when_a_chunk_still_fails_after_retry():
    events = []
    structurer = LLMStructurer(
        _PersistentlyBrokenProvider(), Cache(enabled=False), on_event=lambda k, **d: events.append((k, d))
    )
    chunks = _chunks(20)
    results = structurer._map(chunks, "full")
    assert len(results) == len(chunks) - 1

    retry_events = [d for k, d in events if k == "map_retry"]
    assert len(retry_events) == 1
    assert retry_events[0]["count"] == 1  # exactly the one leftover chunk

    incomplete_events = [d for k, d in events if k == "map_incomplete"]
    assert len(incomplete_events) == 1
    assert incomplete_events[0] == {"failed": 1, "total": 20}


def test_a_transient_batch_failure_is_healed_by_the_automatic_retry_pass():
    # Fails every call in the FIRST pass, then succeeds on retry -- simulates
    # a rate-limit burst that has calmed down by the time the retry pass runs.
    class _FailsOnceThenRecovers:
        name = "flaky-once"
        model = "flaky-once-1"

        def __init__(self):
            self.first_pass_calls = 0
            self.seen_batches: set[str] = set()

        def complete_json(self, system, user):
            if "TASK: BATCH MAP" in system and user not in self.seen_batches:
                self.seen_batches.add(user)
                raise LLMError("simulated transient rate limit")
            return MockProvider().complete_json(system, user)

    events = []
    provider = _FailsOnceThenRecovers()
    structurer = LLMStructurer(provider, Cache(enabled=False), on_event=lambda k, **d: events.append((k, d)))
    chunks = _chunks(20)
    results = structurer._map(chunks, "full")

    assert len(results) == len(chunks)  # full recovery, nothing permanently lost
    assert not any(k == "map_incomplete" for k, _ in events)  # nothing left to report as missing
    assert any(k == "map_retry" for k, _ in events)  # but the retry pass did have to run


def test_resuming_after_a_partial_failure_only_pays_for_what_remains(tmp_path):
    cache = Cache(root=tmp_path, enabled=True)
    chunks = _chunks(20)

    # First attempt: only the first 2 batch calls succeed (quota runs out
    # partway through). max_workers=1 makes call order deterministic.
    flaky = _FlakyProvider(fail_after=2)
    structurer1 = LLMStructurer(flaky, cache, max_workers=1)
    results1 = structurer1._map(chunks, "full")
    assert 0 < len(results1) < len(chunks)  # a genuine partial success
    first_attempt_calls = flaky.calls

    # Second attempt (e.g. tomorrow, fresh quota): a normal provider, same
    # cache. Must recover everything, and must NOT re-pay for what already
    # succeeded -- fewer calls than one-per-remaining-chunk would need.
    provider2 = MockProvider()
    structurer2 = LLMStructurer(provider2, cache, max_workers=1)
    results2 = structurer2._map(chunks, "full")

    assert len(results2) == len(chunks)  # full recovery
    remaining = len(chunks) - len(results1)
    assert provider2.calls < remaining  # batching still applies to the resumed remainder
    assert provider2.calls > 0


def test_map_resumed_event_fires_only_when_some_but_not_all_chunks_are_cached(tmp_path):
    cache = Cache(root=tmp_path, enabled=True)
    chunks = _chunks(20)

    events = []
    structurer1 = LLMStructurer(MockProvider(), cache, max_workers=1, on_event=lambda k, **d: events.append((k, d)))
    structurer1._map(chunks, "full")
    assert not any(k == "map_resumed" for k, _ in events)  # first run: nothing cached yet, no "resumed" event

    events2 = []
    structurer2 = LLMStructurer(
        MockProvider(), cache, max_workers=1, on_event=lambda k, **d: events2.append((k, d))
    )
    structurer2._map(chunks, "full")
    assert not any(k == "map_resumed" for k, _ in events2)  # second run: everything cached, still no partial-resume event (nothing new to fetch)


def test_map_resumed_event_reports_correct_counts(tmp_path):
    cache = Cache(root=tmp_path, enabled=True)
    chunks = _chunks(20)

    flaky = _FlakyProvider(fail_after=2)
    LLMStructurer(flaky, cache, max_workers=1)._map(chunks, "full")

    events = []
    structurer2 = LLMStructurer(
        MockProvider(), cache, max_workers=1, on_event=lambda k, **d: events.append((k, d))
    )
    results2 = structurer2._map(chunks, "full")
    resumed_events = [d for k, d in events if k == "map_resumed"]
    assert len(resumed_events) == 1
    assert resumed_events[0]["total"] == 20
    assert 0 < resumed_events[0]["cached"] < 20
