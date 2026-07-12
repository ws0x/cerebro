from cerebro.convert.markdown import mindmap_to_markdown, write_markdown
from cerebro.ir import MindMap, Node, NodeType, Relationship


def test_title_becomes_h1():
    mm = MindMap(title="My Great Map", root=Node(title="My Great Map", type=NodeType.root))
    text = mindmap_to_markdown(mm)
    assert text.startswith("# My Great Map\n")


def test_children_become_nested_bullets():
    root = Node(title="Root", type=NodeType.root)
    a = root.add("Topic A")
    a.add("Subtopic A1")
    root.add("Topic B")
    mm = MindMap(title="Root", root=root)

    text = mindmap_to_markdown(mm)
    lines = text.splitlines()
    assert "- Topic A" in lines
    assert "  - Subtopic A1" in lines
    assert "- Topic B" in lines


def test_root_itself_is_not_duplicated_as_a_bullet():
    # The root's own title is already the H1 -- it shouldn't also appear as
    # a redundant top-level bullet.
    root = Node(title="Root Title", type=NodeType.root)
    root.add("Only Child")
    mm = MindMap(title="Root Title", root=root)
    text = mindmap_to_markdown(mm)
    assert text.count("Root Title") == 1  # only in the H1


def test_note_renders_as_indented_italic_line_under_its_bullet():
    root = Node(title="Root", type=NodeType.root)
    root.add("Topic", note="An explanatory note.")
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)
    lines = text.splitlines()
    topic_idx = lines.index("- Topic")
    assert lines[topic_idx + 1] == "  *An explanatory note.*"


def test_multiline_note_collapses_to_one_bullet_line():
    root = Node(title="Root", type=NodeType.root)
    root.add("Topic", note="Line one.\nLine two.\n\nLine three.")
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)
    assert "Line one. Line two. Line three." in text
    assert "\n\n" not in text.split("## Relationships")[0].split("- Topic", 1)[1][:60]


def test_no_relationships_section_when_there_are_none():
    root = Node(title="Root", type=NodeType.root)
    root.add("Topic")
    mm = MindMap(title="Root", root=root)
    assert "## Relationships" not in mindmap_to_markdown(mm)


def test_relationships_section_lists_titles_not_raw_ids():
    root = Node(title="Root", type=NodeType.root)
    a = root.add("Concept A")
    b = root.add("Concept B")
    mm = MindMap(title="Root", root=root, relationships=[Relationship(from_id=a.id, to_id=b.id, label="leads to")])
    text = mindmap_to_markdown(mm)
    assert "## Relationships" in text
    assert "- **Concept A** → **Concept B** (leads to)" in text
    assert a.id not in text
    assert b.id not in text


def test_relationship_with_no_label_omits_the_parens():
    root = Node(title="Root", type=NodeType.root)
    a = root.add("A")
    b = root.add("B")
    mm = MindMap(title="Root", root=root, relationships=[Relationship(from_id=a.id, to_id=b.id, label="")])
    text = mindmap_to_markdown(mm)
    assert "- **A** → **B**\n" in text or text.rstrip().endswith("- **A** → **B**")


def test_deep_nesting_indents_correctly():
    root = Node(title="Root", type=NodeType.root)
    n = root
    for i in range(4):
        n = n.add(f"Level {i}")
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)
    assert "      - Level 3" in text  # depth 3 (0-indexed from first child) -> 3 * 2 spaces


def test_write_markdown_creates_parent_dirs_and_writes_utf8(tmp_path):
    root = Node(title="Root", type=NodeType.root)
    root.add("Café ☕ topic")
    mm = MindMap(title="Root", root=root)
    out = write_markdown(mm, tmp_path / "nested" / "dir" / "map.md")
    assert out.exists()
    assert "Café ☕ topic" in out.read_text(encoding="utf-8")


def test_semantic_node_types_get_a_tag_prefix():
    root = Node(title="Root", type=NodeType.root)
    root.add("A definition", type=NodeType.definition)
    root.add("A warning", type=NodeType.warning)
    root.add("A concept", type=NodeType.concept)
    root.add("An example", type=NodeType.example)
    root.add("An insight", type=NodeType.insight)
    root.add("An action", type=NodeType.action)
    root.add("A question", type=NodeType.question)
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)

    assert "- [Definition] A definition" in text
    assert "- [Warning] A warning" in text
    assert "- [Concept] A concept" in text
    assert "- [Example] An example" in text
    assert "- [Insight] An insight" in text
    assert "- [Action] An action" in text
    assert "- [Question] A question" in text


def test_structural_node_types_are_not_tagged():
    root = Node(title="Root", type=NodeType.root)
    root.add("Plain topic", type=NodeType.topic)
    root.add("Plain detail", type=NodeType.detail)
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)

    assert "- Plain topic" in text
    assert "- Plain detail" in text
    assert "[" not in text.split("# Root", 1)[1]


def test_type_tag_survives_alongside_a_note():
    root = Node(title="Root", type=NodeType.root)
    root.add("Tricky part", type=NodeType.warning, note="Watch out for this.")
    mm = MindMap(title="Root", root=root)
    text = mindmap_to_markdown(mm)
    lines = text.splitlines()
    idx = lines.index("- [Warning] Tricky part")
    assert lines[idx + 1] == "  *Watch out for this.*"
