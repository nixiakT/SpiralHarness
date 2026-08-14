"""Role-exact media types shared by feedback contracts and trusted producers."""

SAFE_BENCHMARK_METADATA_MEDIA_TYPE = (
    "application/vnd.spiral-harness.safe-benchmark-metadata.v1+json"
)
EXPLORATION_INPUTS_MEDIA_TYPE = "application/vnd.spiral-harness.exploration-inputs.v1+json"

__all__ = [
    "EXPLORATION_INPUTS_MEDIA_TYPE",
    "SAFE_BENCHMARK_METADATA_MEDIA_TYPE",
]
