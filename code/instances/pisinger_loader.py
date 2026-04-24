"""Loader for the Pisinger 1995 `mcknap` MCKP instance archive.

Pisinger's instance files are plain text with the following layout
(`.in` files in the original archive):

    N                          # number of classes
    M_1                        # items in class 1
    p_{1,1}  c_{1,1}
    p_{1,2}  c_{1,2}
    ...
    M_2
    p_{2,1}  c_{2,1}
    ...
    M_N
    p_{N,1}  c_{N,1}
    ...
    B                          # knapsack capacity

The Pisinger archive is the **classic MCKP** form: every class must
contribute exactly one item. We treat each Pisinger MCKP instance as a
Selective-MCKP instance whose budget happens to be slack enough that
every class is selected at the optimum (the manuscript's §3.8 sanity
check uses this property). We do NOT pad with a dummy zero-cost item
here — that transformation lives in the BISSA adapter (`code/solvers/`).

The Pisinger archive is **not** redistributed in this repository: it
must be fetched from Pisinger's web page and dropped under
`instances/pisinger_1995/`. See `instances/pisinger_1995/README.md` for
the canonical URL and the SHA-256 checksums of every file we have
verified.
"""

from __future__ import annotations

from pathlib import Path

from instances.schema import GENERATOR_VERSION, CorrelationKind, InstanceModel


def parse_pisinger_text(text: str, *, source: str = "pisinger_1995") -> InstanceModel:
    """Parse one Pisinger `.in` file and return an `InstanceModel`.

    Pisinger's MCKP archive does not record a correlation tag in the
    file header, so we set `correlation = uncorrelated` as a neutral
    default; the file's bucket directory under `instances/pisinger_1995/`
    carries the actual correlation in its name.
    """
    tokens = _tokenise(text)
    pos = 0

    N, pos = _read_int(tokens, pos)
    classes: list[list[list[int]]] = []
    M_seen: int | None = None
    for _ in range(N):
        M_i, pos = _read_int(tokens, pos)
        if M_seen is None:
            M_seen = M_i
        if M_i != M_seen:
            raise ValueError(
                f"variable M_i is unsupported by the unified schema "
                f"(saw {M_seen} then {M_i}); split the archive instead"
            )
        items: list[list[int]] = []
        for _j in range(M_i):
            p, pos = _read_int(tokens, pos)
            c, pos = _read_int(tokens, pos)
            items.append([p, c])
        classes.append(items)

    B, _pos = _read_int(tokens, pos)
    if M_seen is None:
        raise ValueError("empty Pisinger instance: no classes parsed")

    return InstanceModel(
        N=N,
        M=M_seen,
        correlation=CorrelationKind.UNCORRELATED,
        f=_recover_f(N, classes, B),
        seed=0,
        B=B,
        items=classes,
        generator_version=f"pisinger_loader+{GENERATOR_VERSION} ({source})",
    )


def load_pisinger_file(path: str | Path) -> InstanceModel:
    """Read and parse one Pisinger `.in` file from disk."""
    path = Path(path)
    return parse_pisinger_text(path.read_text(), source=path.name)


def _tokenise(text: str) -> list[str]:
    return [t for line in text.splitlines() for t in line.split() if not line.startswith("#")]


def _read_int(tokens: list[str], pos: int) -> tuple[int, int]:
    if pos >= len(tokens):
        raise ValueError(f"truncated Pisinger file at token index {pos}")
    return int(tokens[pos]), pos + 1


def _recover_f(N: int, classes: list[list[list[int]]], B: int) -> float:
    mean_cost = sum(c for cls in classes for _p, c in cls) / max(1, sum(len(cls) for cls in classes))
    if mean_cost <= 0:
        return 0.5
    return max(1e-3, B / (N * mean_cost))
