# Synthetic dataset manifest fixture

`manifest.parquet` is a deterministic contract fixture with 1,000 canonical
design/case identities and synthetic run digests, metrics, sizes, and
provenance. It exercises the standalone manifest validator without claiming
that the canonical remote dataset has been produced. Parent run artifacts are
fully verified by `build_manifest`; this compact fixture intentionally does not
ship 1,000 field archives.

`tests/datagen/test_manifest.py` regenerates the logical fixture in memory and
requires byte-for-byte equality.
