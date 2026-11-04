# Galvo Memory Agent Bootstrap

Read `../LUCIDITY.md` before making architectural decisions in this repo.

This repo owns memory and evidence behavior for Galvo-family workflows when
those concerns are separate from GalvoFS. Keep memory/evidence semantics aligned
with GalvoFS receipts where they overlap, but do not turn this repo into the
durable shared filesystem.

For storage substrate work, use the GalvoFS roadmap currently incubating in
`../Femto/ROADMAP.md`.
