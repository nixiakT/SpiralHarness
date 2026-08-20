"""Shared evidence-shape contracts for verified BFCL v2 semantic authority."""

from __future__ import annotations

from enum import StrEnum

from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_pairwise_v4_release_contracts import (
    BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_semantic_release_contracts import (
    BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE,
)


class BfclV4PublicV2SemanticReleaseEvidenceShape(StrEnum):
    """The only semantic evidence shapes admitted to score-bearing execution."""

    LEGACY_SINGLE_PACKET_V1 = "legacy-single-packet-v1"
    CHUNKED_PAIRWISE_COMPOSITE_V4 = "chunked-pairwise-composite-v4"


def bfcl_v4_public_v2_semantic_release_media_type(
    shape: BfclV4PublicV2SemanticReleaseEvidenceShape,
) -> str:
    """Return the one release media type bound to an evidence shape."""

    return {
        BfclV4PublicV2SemanticReleaseEvidenceShape.LEGACY_SINGLE_PACKET_V1: (
            BFCL_V4_PUBLIC_V2_SEMANTIC_DEVELOPMENT_RELEASE_MEDIA_TYPE
        ),
        BfclV4PublicV2SemanticReleaseEvidenceShape.CHUNKED_PAIRWISE_COMPOSITE_V4: (
            BFCL_V4_PUBLIC_V2_PAIRWISE_V4_DEVELOPMENT_RELEASE_MEDIA_TYPE
        ),
    }[shape]


__all__ = [
    "BfclV4PublicV2SemanticReleaseEvidenceShape",
    "bfcl_v4_public_v2_semantic_release_media_type",
]
