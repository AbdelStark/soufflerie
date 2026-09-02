# Canonical dataset evidence

This directory is reserved for the checked, digest-bound outputs of the
authenticated `canonical-lhs-v1` sweep. A release evidence update must include
the published dataset metadata and statistics, the terminal sweep summary, and
an evidence-bounded cost calculation. Large run archives remain in the
immutable Modal volume and are named by full digest rather than copied into
Git.

The synthetic manifest under `tests/fixtures/` is contract data and must never
be copied here or presented as canonical run evidence.
