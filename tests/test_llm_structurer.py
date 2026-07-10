from cerebro.cache import Cache
from cerebro.ir import MindMap, Node, NodeType, Relationship
from cerebro.llm.providers import MockProvider
from cerebro.structure.llm import LLMStructurer, chunk_transcript, link_relationships
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


class CustomMockProvider(MockProvider):
    def __init__(self, response: dict):
        super().__init__()
        self.response = response
        self.last_system = None
        self.last_user = None

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return self.response


def test_link_relationships_works_on_a_hand_built_multi_branch_tree():
    root = Node(title="Course", type=NodeType.root)
    video_a = root.add("Video A", type=NodeType.topic)
    video_a.add("Concept One", type=NodeType.concept)
    video_b = root.add("Video B", type=NodeType.topic)
    video_b.add("Concept Two", type=NodeType.concept)
    mm = MindMap(title="Course", root=root, level="expert")

    # Propose link from node 1 (Concept One) to node 3 (Concept Two)
    # nodes listing is:
    # 0: Video A, 1: Concept One, 2: Video B, 3: Concept Two
    provider = CustomMockProvider({"relationships": [{"from": 1, "to": 3, "label": "builds on"}]})
    link_relationships(mm, provider, Cache(enabled=False), cross_video=True)

    assert len(mm.relationships) == 1
    assert mm.relationships[0].from_id == mm.root.children[0].children[0].id
    assert mm.relationships[0].to_id == mm.root.children[1].children[0].id

    # Verify that the correct system prompt was used and user content included "video"
    assert "CROSS-VIDEO LINKING" in provider.last_system
    import json
    user_data = json.loads(provider.last_user)
    assert user_data[1]["video"] == "Video A"
    assert user_data[3]["video"] == "Video B"


def test_link_relationships_rejects_same_branch_links():
    root = Node(title="Course", type=NodeType.root)
    video_a = root.add("Video A", type=NodeType.topic)
    video_a.add("Concept One", type=NodeType.concept)
    mm = MindMap(title="Course", root=root, level="expert")

    # Propose link from 0 (Video A) to 1 (Concept One), which are in the same branch
    provider = CustomMockProvider({"relationships": [{"from": 0, "to": 1, "label": "invalid same branch"}]})
    link_relationships(mm, provider, Cache(enabled=False))

    assert len(mm.relationships) == 0


def test_link_relationships_discards_duplicates():
    root = Node(title="Course", type=NodeType.root)
    video_a = root.add("Video A", type=NodeType.topic)
    concept_one = video_a.add("Concept One", type=NodeType.concept)
    video_b = root.add("Video B", type=NodeType.topic)
    concept_two = video_b.add("Concept Two", type=NodeType.concept)
    
    # Pre-add relationship in mind map
    mm = MindMap(title="Course", root=root, level="expert")
    mm.relationships.append(Relationship(from_id=concept_one.id, to_id=concept_two.id, label="already exists"))

    # Mock provider proposes the same relationship (node 1 to node 3)
    provider = CustomMockProvider({"relationships": [{"from": 1, "to": 3, "label": "duplicate"}]})
    link_relationships(mm, provider, Cache(enabled=False))

    # Should still only have the pre-existing relationship
    assert len(mm.relationships) == 1
    assert mm.relationships[0].label == "already exists"


def test_link_relationships_skips_trivially_small_trees():
    root = Node(title="Tiny", type=NodeType.root)
    root.add("Only Child")
    mm = MindMap(title="Tiny", root=root, level="expert")

    provider = MockProvider()
    link_relationships(mm, provider, Cache(enabled=False))

    assert mm.relationships == []
    assert provider.calls == 0  # never even called the model for 2 nodes


def test_cache_avoids_recompute(tmp_path):
    cache = Cache(root=tmp_path / "c", enabled=True)
    provider = MockProvider()
    structurer = LLMStructurer(provider, cache=cache)

    structurer.structure(_transcript(), level="full")
    calls_after_first = provider.calls
    assert calls_after_first > 0

    structurer.structure(_transcript(), level="full")
    assert provider.calls == calls_after_first  # fully served from cache
