# XMind theme reference

`DEFAULT_MAP_TEMPLATE.xmind` is a byte-for-byte backup of the reference map
`convert/xmind.py`'s visual theme was copied from — a real XMind Zen "Dawn"
theme (multi-line branch colors, rounded topics, clockwise radial layout).

It exists purely as a durable backup: `xmind.py` doesn't read this file at
runtime, and never has — the theme's colors/fonts/shapes were copied into
plain Python constants (`_THEME`, `_EXTENSIONS`, `_STRUCTURE_CLASS`) once,
by hand, from this file's `content.json`. Keeping this copy in the repo
just means that transcription can always be re-verified (or redone from
scratch, if the theme ever needs to change) without depending on the
original file surviving on its owner's machine.

`tests/test_xmind.py::test_embedded_theme_is_byte_identical_to_the_preserved_reference_template`
compares `xmind.py`'s live theme constant against this file on every test
run, so the two can never silently drift apart.
