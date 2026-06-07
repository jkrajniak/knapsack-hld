"""Pydantic schema for Selective-MCKP benchmark instances.

The on-disk format is JSON (optionally gzipped). One file = one instance.
The schema is a faithful translation of the manuscript's notation:

    N           Number of classes
    M           Number of items per class (uniform across classes)
    correlation Correlation class between profits and costs
    f           Budget tightness factor (B = round(f · N · mean(c)))
    seed        Master seed used by the generator
    B           Total budget (computed from f, N, items)
    items       Length-N list of length-M lists of [profit, cost] pairs

Constraints:
- Profits and costs are non-negative integers in [1, R] where R is the
  generator's amplitude parameter (default 1000).
- Profits and costs MUST round-trip bit-exactly for a given seed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

GENERATOR_VERSION = "0.1.0"


class CorrelationKind(StrEnum):
    """Profit/cost correlation classes (Pisinger 1995, Martello & Toth 1990).

    The first four are classical correlation labels reused by the project's
    own generator. SUBSET_SUM, SIMILAR_WEIGHTS, and UNCORRELATED_WITH_SKIP
    correspond one-to-one to Pisinger 1995 instance types 4, 5, and 6 and
    are produced only by `pisinger_generator`.
    """

    UNCORRELATED = "uncorrelated"
    WEAKLY = "weakly"
    STRONGLY = "strongly"
    INVERSELY_STRONGLY = "inversely_strongly"
    SUBSET_SUM = "subset_sum"
    SIMILAR_WEIGHTS = "similar_weights"
    UNCORRELATED_WITH_SKIP = "uncorrelated_with_skip"


class InstanceModel(BaseModel):
    """A single Selective-MCKP benchmark instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    N: PositiveInt = Field(..., description="Number of classes")
    M: PositiveInt = Field(..., description="Items per class (uniform)")
    correlation: CorrelationKind
    f: float = Field(..., gt=0.0, description="Budget tightness factor B = f·N·mean(c)")
    seed: NonNegativeInt = Field(..., description="Master seed for the generator")
    B: PositiveInt = Field(..., description="Total budget (integer)")
    items: list[list[list[NonNegativeInt]]] = Field(
        ...,
        description="items[i][j] = [profit, cost] for item j of class i",
    )
    generator_version: str = Field(default=GENERATOR_VERSION)

    @model_validator(mode="after")
    def _check_shapes(self) -> "InstanceModel":
        if len(self.items) != self.N:
            raise ValueError(f"items has {len(self.items)} classes but N={self.N}")
        for i, cls in enumerate(self.items):
            if len(cls) != self.M:
                raise ValueError(f"class {i} has {len(cls)} items but M={self.M}")
            for j, pc in enumerate(cls):
                if len(pc) != 2:
                    raise ValueError(f"items[{i}][{j}] is not a [profit, cost] pair")
        return self
