# XMind theme reference

Two byte-for-byte backups of the reference maps `convert/xmind.py`'s two
visual themes were copied from:

- **`DEFAULT_MAP_TEMPLATE.xmind`** — the "Dawn" theme (multi-line branch
  colors, rounded topics, clockwise radial layout) used for video/PDF/
  article/local-file content maps.
- **`TREE_MAP_TEMPLATE.xmind`** — the "Hawaii" theme (cooler blue/teal/amber
  palette, right-hand logic-chart layout) used for `cerebro tree`
  folder-structure maps specifically, selected automatically whenever
  `MindMap.level == "structure"`.

Both exist purely as durable backups: `xmind.py` doesn't read either file at
runtime, and never has — each theme's colors/fonts/shapes were copied into
plain Python constants (`_THEME`/`_EXTENSIONS`/`_STRUCTURE_CLASS` for Dawn,
`_TREE_THEME`/`_TREE_EXTENSIONS`/`_TREE_STRUCTURE_CLASS` for Hawaii) once, by
hand, from each file's own `content.json`. Keeping these copies in the repo
just means that transcription can always be re-verified (or redone from
scratch, if a theme ever needs to change) without depending on either
original file surviving on its owner's machine.

`tests/test_xmind.py`'s
`test_embedded_theme_is_byte_identical_to_the_preserved_reference_template`
and `test_tree_theme_is_byte_identical_to_the_preserved_reference_template`
compare `xmind.py`'s live theme constants against these two files on every
test run, so neither pair can ever silently drift apart.
