"""Generate or byte-check the committed synthetic CPU smoke model resources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from soufflerie.schemas import canonical_json_bytes, canonical_sha256, sha256_bytes
from soufflerie.surrogate.bundle import (
    EXPECTED_MODEL_TENSORS,
    MODEL_ARCHITECTURE_NAME,
    MODEL_CARD_NAME,
    MODEL_METADATA_NAME,
    MODEL_PREPROCESSING_NAME,
    ModelCardGate,
    ModelCardMetadata,
    build_model_bundle,
)
from soufflerie.surrogate.bundled import (
    BUNDLED_COMPRESSED_WEIGHTS_NAME,
    BUNDLED_RESOURCE_NAME,
    BundledModelResource,
    SmokeDatasetParent,
    rendered_bundled_resource_bytes,
)
from soufflerie.surrogate.preprocessing import (
    MODEL_CELL_COUNT,
    MODEL_SPATIAL_SHAPE,
    OutputChannelStatistics,
    OutputNormalizationStatistics,
    PreprocessingStatistics,
)

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "soufflerie" / "resources" / "model"
BUNDLE_SOURCE_REVISION = "509115ce8cb50d224eb3705f2b276ab3da41e647"


def _weights() -> dict[str, npt.NDArray[np.float32]]:
    weights = {
        descriptor.name: np.zeros(descriptor.shape, dtype=np.float32)
        for descriptor in EXPECTED_MODEL_TENSORS
    }
    weights["core.decoder_net.final_layer.linear.bias"][:] = np.array(
        [0.25, -0.5, 0.75], dtype=np.float32
    )
    weights["cd_head.4.bias"][:] = np.float32(0.125)
    return weights


def _preprocessing(dataset_id: str) -> PreprocessingStatistics:
    channel = OutputChannelStatistics(
        mean=0.0,
        raw_standard_deviation=1.0,
        standard_deviation=1.0,
        floored=False,
    )
    return PreprocessingStatistics(
        dataset_id=dataset_id,
        training_case_count=1,
        training_cell_count=MODEL_CELL_COUNT,
        outputs=OutputNormalizationStatistics(
            u_mean=channel,
            v_mean=channel,
            rho_delta=channel,
        ),
    )


def _card() -> ModelCardMetadata:
    return ModelCardMetadata(
        display_name="Soufflerie synthetic CPU smoke FNO",
        summary=(
            "A deterministic untrained fixture for installed-wheel bundle and CPU inference smoke."
        ),
        intended_uses=("Installed-wheel integrity and CPU inference contract testing.",),
        limitations=(
            "Synthetic zero-weight fixture with fixed output biases; not flow-accuracy evidence.",
            "Do not use this fixture for scientific prediction or release-quality claims.",
        ),
        gates=(
            ModelCardGate(
                name="Scientific validation",
                status="not_evaluated",
                threshold="Requires a separately trained and RFC-0008-validated model",
            ),
        ),
    )


def _gzip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as stream:
        stream.write(content)
    return buffer.getvalue()


def generate_resource_documents() -> dict[str, bytes]:
    fixture_parent = SmokeDatasetParent()
    dataset_sha256 = fixture_parent.sha256
    experiment_id = canonical_sha256(
        {
            "kind": "synthetic-cpu-smoke-experiment-v1",
            "dataset_sha256": dataset_sha256,
            "architecture": "fno2d-v1",
            "seed": 0,
        }
    )[:20]
    bundle = build_model_bundle(
        weights=_weights(),
        preprocessing=_preprocessing(dataset_sha256[:20]),
        dataset_sha256=dataset_sha256,
        experiment_id=experiment_id,
        seed=0,
        selected_epoch=1,
        code_revision=BUNDLE_SOURCE_REVISION,
        lock_digest=sha256_bytes((PROJECT_ROOT / "uv.lock").read_bytes()),
        model_card=_card(),
    )
    compressed_weights = _gzip(bundle.weights_bytes)
    fields = np.empty((1, 3, *MODEL_SPATIAL_SHAPE), dtype=np.float32)
    fields[:, 0] = np.float32(0.25)
    fields[:, 1] = np.float32(-0.5)
    fields[:, 2] = np.float32(0.75)
    cd = np.array([0.125], dtype=np.float32)
    metadata_bytes = canonical_json_bytes(bundle.metadata)
    descriptor = BundledModelResource(
        model=bundle.reference,
        bundle_metadata_sha256=sha256_bytes(metadata_bytes),
        fixture_parent=fixture_parent,
        fixture_parent_sha256=dataset_sha256,
        compressed_weights_sha256=sha256_bytes(compressed_weights),
        compressed_weights_bytes=len(compressed_weights),
        uncompressed_weights_bytes=len(bundle.weights_bytes),
        expected_fields_sha256=hashlib.sha256(fields.tobytes(order="C")).hexdigest(),
        expected_cd_sha256=hashlib.sha256(cd.tobytes(order="C")).hexdigest(),
    )
    return {
        BUNDLED_RESOURCE_NAME: rendered_bundled_resource_bytes(descriptor),
        MODEL_METADATA_NAME: metadata_bytes,
        MODEL_PREPROCESSING_NAME: canonical_json_bytes(bundle.preprocessing),
        MODEL_ARCHITECTURE_NAME: canonical_json_bytes(bundle.architecture),
        MODEL_CARD_NAME: bundle.model_card_markdown.encode("utf-8"),
        BUNDLED_COMPRESSED_WEIGHTS_NAME: compressed_weights,
    }


def check_resources(output: Path, documents: dict[str, bytes]) -> tuple[str, ...]:
    if not output.is_dir():
        return (f"missing generated resource directory: {output}",)
    observed = {path.name for path in output.iterdir()}
    expected = set(documents)
    errors = (
        [f"resource members differ: expected {sorted(expected)}, got {sorted(observed)}"]
        if observed != expected
        else []
    )
    for name, expected_content in documents.items():
        path = output / name
        if not path.is_file():
            continue
        content = path.read_bytes()
        if content != expected_content:
            errors.append(f"generated resource differs: {path}")
    return tuple(errors)


def write_resources(output: Path, documents: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    extras = {path.name for path in output.iterdir()} - set(documents)
    if extras:
        raise RuntimeError(f"refusing to overwrite directory with extra members: {sorted(extras)}")
    for name, content in documents.items():
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-", dir=output)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, output / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    documents = generate_resource_documents()
    if args.check:
        errors = check_resources(args.output, documents)
        if errors:
            for error in errors:
                print(f"bundled model error: {error}")
            return 1
        print(
            "bundled_model=PASS "
            f"members={len(documents)} compressed_bytes="
            f"{len(documents[BUNDLED_COMPRESSED_WEIGHTS_NAME])}"
        )
        return 0
    write_resources(args.output, documents)
    print(f"wrote {len(documents)} bundled model resources to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
