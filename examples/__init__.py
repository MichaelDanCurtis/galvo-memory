"""Galvo memory examples.

Operator-facing smokes and walkthroughs. Cycle 1 ships a single bash
smoke (``sidecar_demo.sh``) that drives the FastAPI sidecar through
the six acceptance gates from ``memory/docs/PHASE-2-PLAN.md``.

Python entrypoints may land in cycle 2 (e.g. a notebook-style walkthrough
or a richer scoring-signal demo); this ``__init__.py`` reserves the
directory as an importable package so those can drop in without a
follow-up "make this a package" diff.
"""
