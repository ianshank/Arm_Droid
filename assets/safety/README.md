# Safety guard data

This directory ships inside the installed wheel via the
`[tool.hatch.build.targets.wheel.force-include]` mapping declared in
`pyproject.toml` so the Asimov-style safety guard introduced in Phase F
(ADR-0009) finds its rule corpus at a stable, configurable path
(`ArmSafetyConfig.neiss_corpus_path`).

## Layout (populated by Phase F)

```
assets/safety/
├── README.md            # this file
└── neiss_corpus.json    # curated NEISS injury subset (Phase F)
```

## Phase A status

Phase A (the Gemini Robotics ER 1.6 foundation scaffolding) reserves
the wheel asset slot and declares the `ArmSafetyConfig.neiss_corpus_path`
field; the corpus JSON itself is not yet shipped. The
`ArmSafetyConfig.asimov_rule_set` discriminator defaults to `"off"` so
no consumer reads the path until an operator opts in.

A Phase F regression test (`tests/regression/test_neiss_corpus_packaged.py`,
to be added in that PR) will assert that an installed wheel resolves
`cfg.arm_safety.neiss_corpus_path` to a real file. Until then this
README keeps the directory present so the `force-include` mapping does
not break editable installs.

## NEISS attribution

The NEISS database (National Electronic Injury Surveillance System) is
maintained by the U.S. Consumer Product Safety Commission and published
under public-domain terms. Phase F bundles a curated subset relevant to
robotics fragile-object handling; the per-record license review is
tracked as open question 3 in
`/root/.claude/plans/as-per-the-research-radiant-nova.md`.
