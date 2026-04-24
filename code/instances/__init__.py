"""Selective-MCKP benchmark instances: schema, generator, IO."""

from instances.generator import generate_instance
from instances.io import instance_id, load_instance, save_instance
from instances.schema import (
    GENERATOR_VERSION,
    CorrelationKind,
    InstanceModel,
)

__all__ = [
    "GENERATOR_VERSION",
    "CorrelationKind",
    "InstanceModel",
    "generate_instance",
    "instance_id",
    "load_instance",
    "save_instance",
]
