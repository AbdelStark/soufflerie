from __future__ import annotations

import pytest

from soufflerie.config import MAX_CONFIG_BYTES, ServiceConfig, parse_config
from soufflerie.errors import ConfigurationError, SchemaVersionError

BASE_SERVICE = """
schema_version: 1
model_id: "00000000000000000001"
dataset_id: "00000000000000000000"
report_id: "00000000000000000006"
"""


def test_equivalent_yaml_formatting_has_one_digest() -> None:
    first = parse_config(BASE_SERVICE, ServiceConfig)
    second = parse_config(
        """
        # Mapping order and comments are not identity inputs.
        report_id: "00000000000000000006"
        dataset_id: "00000000000000000000"
        model_id: "00000000000000000001"
        schema_version: 1
        """,
        ServiceConfig,
    )
    assert first == second
    assert first.config_digest == second.config_digest


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        ("unknown: 1\n", "Extra inputs are not permitted"),
        ('port: "8000"\n', "valid integer"),
        ("solve_enabled: yes\n", "valid boolean"),
        ("port: .nan\n", "NaN or infinity"),
        ("host: ${SERVICE_HOST}\n", "environment interpolation"),
    ],
)
def test_unknown_keys_coercions_nonfinite_and_environment_references_fail(
    suffix: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        parse_config(BASE_SERVICE + suffix, ServiceConfig)


def test_duplicate_keys_anchors_and_aliases_fail() -> None:
    with pytest.raises(ConfigurationError, match="duplicate configuration key"):
        parse_config(BASE_SERVICE + "port: 8000\nport: 8001\n", ServiceConfig)

    anchored = """
    schema_version: 1
    model_id: &model "00000000000000000001"
    dataset_id: "00000000000000000000"
    report_id: *model
    """
    with pytest.raises(ConfigurationError, match="anchors are forbidden"):
        parse_config(anchored, ServiceConfig)


def test_unsafe_tags_multiple_documents_and_non_mapping_roots_fail() -> None:
    unsafe = BASE_SERVICE + "payload: !!python/object:builtins.object {}\n"
    with pytest.raises(ConfigurationError, match="single-document safe YAML"):
        parse_config(unsafe, ServiceConfig)
    with pytest.raises(ConfigurationError, match="single-document safe YAML"):
        parse_config(BASE_SERVICE + "---\n{}\n", ServiceConfig)
    with pytest.raises(ConfigurationError, match="root must be a YAML object"):
        parse_config("[1, 2, 3]", ServiceConfig)


def test_size_utf8_and_schema_version_boundaries_are_typed() -> None:
    with pytest.raises(ConfigurationError, match="exceeds"):
        parse_config(b"x" * (MAX_CONFIG_BYTES + 1), ServiceConfig)
    with pytest.raises(ConfigurationError, match="valid UTF-8"):
        parse_config(b"\xff", ServiceConfig)
    with pytest.raises(SchemaVersionError):
        parse_config(BASE_SERVICE.replace("schema_version: 1", "schema_version: 2"), ServiceConfig)
