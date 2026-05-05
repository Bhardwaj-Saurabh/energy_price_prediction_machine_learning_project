"""Identifier for a specific trained model artifact."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """An opaque identifier for a registered, trained model.

    The string format is decided by the model registry adapter — typically
    ``"<model_name>/<version_number>"`` for MLflow — so the domain stays
    decoupled from the registry's naming scheme. Domain code only requires
    that the value is a non-empty string so every forecast, model card,
    and monitoring record carries a meaningful reference.

    Validation here covers the failure modes that matter at the boundary:
    a non-string sneaking through (JSON / YAML payloads) and an empty
    string (silent dropping of the lineage signal).
    """

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError(f"ModelVersion must be str, got {type(self.value).__name__}")
        if not self.value:
            raise ValueError("ModelVersion must be non-empty")
