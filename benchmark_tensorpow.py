#!/usr/bin/env python3
"""Reproducible benchmark driver for the tensorpow manuscript.

The script compares the virtual-block implementation in ``tensorpow`` with a
brute-force Kronecker construction wherever the latter fits within a user-set
memory budget.  It benchmarks both the two-term example used in the MSc thesis
and a genuine three-term, composite-hypothesis-type expression.

Install first:
    python -m pip install tensorpow==0.3.0

Typical run on a 32 GiB Linux workstation:
    python benchmark_tensorpow.py --output benchmark_results \
        --n-values 2:30 --repeats 3 --cold-repeats 1 --threads 1 --direct-memory-gib 20

The output consists of CSV and JSON files.  Cold timings are measured in fresh
Python processes; warm timings reuse one TensorPowerCalculator instance.
This script supports resuming. It records results continuously; if cancelled midway, 
re-running with the same --output will pick up where it left off.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# These variables must be set before importing NumPy/SciPy in each process.
def set_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value


def parse_n_values(spec: str) -> list[int]:
    """Parse ``2:26``, ``2:26:2``, or ``2,3,5,8``."""
    spec = spec.strip()
    if not spec:
        raise ValueError("empty n-value specification")
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("range syntax is START:STOP or START:STOP:STEP")
        start, stop = parts[:2]
        step = parts[2] if len(parts) == 3 else 1
        if step <= 0 or stop < start:
            raise ValueError("require STEP > 0 and STOP >= START")
        return list(range(start, stop + 1, step))
    values = sorted({int(x) for x in spec.split(",")})
    if not values or values[0] < 1:
        raise ValueError("all tensor powers must be positive")
    return values


def density_matrices(seed: int, d: int = 3) -> tuple[Any, Any, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(3):
        g = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = g @ g.conj().T
        rho /= np.trace(rho)
        matrices.append(np.asarray(rho, dtype=np.complex128))
    return tuple(matrices)  # type: ignore[return-value]


def benchmark_cases(seed: int) -> dict[str, tuple[list[Any], list[float]]]:
    a, b, c = density_matrices(seed)
    return {
        "two_term": ([a, b], [0.5, -0.5]),
        "three_term": ([a, b, c], [0.25, 0.25, -0.5]),
    }


def kron_power(matrix: Any, n: int) -> Any:
    import numpy as np

    result = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for _ in range(n):
        result = np.kron(result, matrix)
    return result


def direct_trace_norm(matrices: Sequence[Any], coeffs: Sequence[float], n: int) -> float:
    import numpy as np

    total = None
    for coefficient, matrix in zip(coeffs, matrices, strict=True):
        term = coefficient * kron_power(matrix, n)
        total = term if total is None else total + term
    assert total is not None
    singular_values = np.linalg.svd(total, compute_uv=False)
    return float(np.sum(singular_values, dtype=np.float64))


def estimated_direct_bytes(d: int, n: int) -> int:
    """Conservative final-plus-temporary complex128 storage estimate."""
    order = d**n
    return 2 * order * order * 16


def median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    return statistics.median(abs(x - median) for x in values)


def numpy_configuration() -> str:
    import numpy as np

    output = io.StringIO()
    try:
        # NumPy 2.x supports a machine-readable mode, but text is more portable.
        from contextlib import redirect_stdout

        with redirect_stdout(output):
            np.show_config()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"Could not obtain NumPy configuration: {exc!r}"
    return output.getvalue()


def software_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


@dataclass
class ResultRow:
    case: str
    n: int
    method: str
    phase: str
    repeat: int
    seconds: float
    value: float
    estimated_direct_bytes: int

    abs_residual: float | None = None
    rel_residual: float | None = None


def cold_worker(args: argparse.Namespace) -> int:
    set_thread_environment(args.threads)
    start = time.perf_counter()
    from tensorpow import TensorPowerCalculator

    cases = benchmark_cases(args.seed)
    matrices, coeffs = cases[args.case]
    calculator = TensorPowerCalculator()
    value = float(
        calculator.schatten_p_norm_weighted(
            matrices, n=args.n, p=1, coeffs=coeffs
        )
    )
    payload = {
        "seconds": time.perf_counter() - start,
        "value": value,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_cold_process(
    script: Path, case: str, n: int, seed: int, threads: int
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--case",
        case,
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"cold worker produced no output; stderr={completed.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "could not parse cold-worker output; "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        ) from exc


def summarize(rows: Sequence[ResultRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[ResultRow]] = {}
    for row in rows:
        groups.setdefault((row.case, row.n, row.method, row.phase), []).append(row)

    summary: list[dict[str, Any]] = []
    for (case, n, method, phase), group in sorted(groups.items()):
        times = [row.seconds for row in group]
        summary.append(
            {
                "case": case,
                "n": n,
                "method": method,
                "phase": phase,
                "count": len(times),
                "median_seconds": statistics.median(times),
                "mad_seconds": median_absolute_deviation(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "value": group[-1].value,
                "abs_residual": next(
                    (row.abs_residual for row in group if row.abs_residual is not None),
                    None,
                ),
                "rel_residual": next(
                    (row.rel_residual for row in group if row.rel_residual is not None),
                    None,
                ),
            }
        )
    return summary


def main(args: argparse.Namespace) -> int:
    set_thread_environment(args.threads)
    import numpy as np
    import scipy
    from tensorpow import TensorPowerCalculator

    n_values = parse_n_values(args.n_values)
    output_stem = Path(args.output).expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    cases = benchmark_cases(args.seed)

    raw_csv = output_stem.with_name(output_stem.name + "_raw.csv")
    summary_json = output_stem.with_name(output_stem.name + "_summary.json")
    metadata_json = output_stem.with_name(output_stem.name + "_metadata.json")
    matrices_npz = output_stem.with_name(output_stem.name + "_matrices.npz")

    completed_tasks = set()
    rows: list[ResultRow] = []

    if raw_csv.exists():
        print(f"Resuming from existing raw CSV: {raw_csv}", flush=True)
        with raw_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for d in reader:
                abs_res = d.get("abs_residual", "")
                rel_res = d.get("rel_residual", "")
                row = ResultRow(
                    case=d["case"],
                    n=int(d["n"]),
                    method=d["method"],
                    phase=d["phase"],
                    repeat=int(d["repeat"]),
                    seconds=float(d["seconds"]),
                    value=float(d["value"]),
                    estimated_direct_bytes=int(d["estimated_direct_bytes"]),
                    abs_residual=float(abs_res) if abs_res else None,
                    rel_residual=float(rel_res) if rel_res else None,
                )
                rows.append(row)
                completed_tasks.add((row.case, row.n, row.method, row.phase, row.repeat))

    calculator_start = time.perf_counter()
    calculator = TensorPowerCalculator()
    calculator_init_seconds = time.perf_counter() - calculator_start

    def get_metadata():
        return {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": sys.argv,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                )
            },
            "versions": {
                "tensorpow": software_version("tensorpow"),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "numpy_configuration": numpy_configuration(),
            "seed": args.seed,
            "n_values": n_values,
            "cases": {
                name: {"coefficients": coeffs, "dimension": matrices[0].shape[0]}
                for name, (matrices, coeffs) in cases.items()
            },
            "repeats": args.repeats,
            "warmups": args.warmups,
            "cold_repeats": args.cold_repeats,
            "direct_repeats": args.direct_repeats,
            "direct_warmups": args.direct_warmups,
            "direct_memory_gib": args.direct_memory_gib,
            "direct_max_n": args.direct_max_n,
            "calculator_init_seconds": calculator_init_seconds,
            "max_rss_note": (
                "ru_maxrss is converted from KiB on non-macOS systems and treated "
                "as bytes on macOS"
            ),
            "output_files": {
                "raw_csv": str(raw_csv),
                "summary_json": str(summary_json),
                "matrices_npz": str(matrices_npz),
            },
        }

    def append_row(r: ResultRow):
        write_header = not raw_csv.exists()
        with raw_csv.open("a", newline="", encoding="utf-8") as handle:
            fieldnames = list(ResultRow.__annotations__)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(r))
        rows.append(r)

        # Update JSON files after each row to ensure state is saved
        summary_json.write_text(
            json.dumps(summarize(rows), indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata_json.write_text(
            json.dumps(get_metadata(), indent=2, sort_keys=True), encoding="utf-8"
        )

    # Write initial metadata and save matrices immediately
    if not matrices_npz.exists():
        np.savez_compressed(
            matrices_npz,
            seed=np.array(args.seed),
            two_term_coeffs=np.array(cases["two_term"][1]),
            three_term_coeffs=np.array(cases["three_term"][1]),
            rho_a=cases["three_term"][0][0],
            rho_b=cases["three_term"][0][1],
            rho_c=cases["three_term"][0][2],
        )
    if not metadata_json.exists():
        metadata_json.write_text(
            json.dumps(get_metadata(), indent=2, sort_keys=True), encoding="utf-8"
        )


    memory_limit = int(args.direct_memory_gib * (1024**3))

    for case_name, (matrices, coeffs) in cases.items():
        for n in n_values:
            estimate = estimated_direct_bytes(matrices[0].shape[0], n)
            
            current_repeats = 0 if n >= 27 else args.repeats
            current_cold_repeats = 1 if n >= 27 else args.cold_repeats
            current_warmups = 0 if n >= 27 else args.warmups

            # --- Virtual Block Warm ---
            if current_repeats > 0:
                skip_warm = all(
                    (case_name, n, "virtual_block", "warm", repeat) in completed_tasks
                    for repeat in range(1, current_repeats + 1)
                )

                if skip_warm:
                    print(f"[{case_name}] n={n}: block warm benchmark (already done)", flush=True)
                else:
                    print(f"[{case_name}] n={n}: block warm benchmark", flush=True)
                    # Warmups
                    for _ in range(current_warmups):
                        calculator.schatten_p_norm_weighted(matrices, n=n, p=1, coeffs=coeffs)

                    for repeat in range(1, current_repeats + 1):
                        if (case_name, n, "virtual_block", "warm", repeat) in completed_tasks:
                            continue
                        
                        start = time.perf_counter()
                        block_value = float(
                            calculator.schatten_p_norm_weighted(matrices, n=n, p=1, coeffs=coeffs)
                        )
                        seconds = time.perf_counter() - start

                        row = ResultRow(
                            case=case_name,
                            n=n,
                            method="virtual_block",
                            phase="warm",
                            repeat=repeat,
                            seconds=seconds,
                            value=block_value,
                            estimated_direct_bytes=estimate,
                        )
                        append_row(row)
                        completed_tasks.add((case_name, n, "virtual_block", "warm", repeat))

            # --- Virtual Block Cold ---
            if current_cold_repeats > 0:
                skip_cold = all(
                    (case_name, n, "virtual_block", "cold_process", repeat) in completed_tasks
                    for repeat in range(1, current_cold_repeats + 1)
                )
                if skip_cold:
                    print(f"[{case_name}] n={n}: block cold benchmark (already done)", flush=True)
                else:
                    print(f"[{case_name}] n={n}: block cold benchmark", flush=True)
                    for repeat in range(1, current_cold_repeats + 1):
                        if (case_name, n, "virtual_block", "cold_process", repeat) in completed_tasks:
                            continue

                        cold = run_cold_process(script, case_name, n, args.seed, args.threads)
                        row = ResultRow(
                            case=case_name,
                            n=n,
                            method="virtual_block",
                            phase="cold_process",
                            repeat=repeat,
                            seconds=float(cold["seconds"]),
                            value=float(cold["value"]),
                            estimated_direct_bytes=estimate,
                        )
                        append_row(row)
                        completed_tasks.add((case_name, n, "virtual_block", "cold_process", repeat))

            # --- Direct Kronecker ---
            if estimate <= memory_limit and n <= args.direct_max_n:
                skip_direct = all(
                    (case_name, n, "direct_kronecker", "warm", repeat) in completed_tasks
                    for repeat in range(1, args.direct_repeats + 1)
                )
                if skip_direct:
                    print(f"[{case_name}] n={n}: direct benchmark (already done)", flush=True)
                else:
                    print(
                        f"[{case_name}] n={n}: direct benchmark "
                        f"(estimated working storage {estimate / 1024**3:.2f} GiB)",
                        flush=True,
                    )
                    # Warmups
                    for _ in range(args.direct_warmups):
                        direct_trace_norm(matrices, coeffs, n)

                    for repeat in range(1, args.direct_repeats + 1):
                        if (case_name, n, "direct_kronecker", "warm", repeat) in completed_tasks:
                            continue

                        start = time.perf_counter()
                        direct_value = direct_trace_norm(matrices, coeffs, n)
                        seconds = time.perf_counter() - start

                        # Retrieve block_value to calculate residuals
                        block_value = None
                        for r in rows:
                            if r.case == case_name and r.n == n and r.method == "virtual_block":
                                block_value = r.value
                                break

                        abs_residual = None
                        rel_residual = None
                        if block_value is not None:
                            abs_residual = abs(block_value - direct_value)
                            rel_residual = abs_residual / max(abs(direct_value), np.finfo(float).tiny)

                        row = ResultRow(
                            case=case_name,
                            n=n,
                            method="direct_kronecker",
                            phase="warm",
                            repeat=repeat,
                            seconds=seconds,
                            value=direct_value,
                            estimated_direct_bytes=estimate,
                            abs_residual=abs_residual,
                            rel_residual=rel_residual,
                        )
                        append_row(row)
                        completed_tasks.add((case_name, n, "direct_kronecker", "warm", repeat))

                        # In-memory update for past virtual_block rows' residuals, so summary JSON is complete
                        for r in rows:
                            if r.case == case_name and r.n == n and r.method == "virtual_block" and r.abs_residual is None:
                                r.abs_residual = abs_residual
                                r.rel_residual = rel_residual
                        
                        summary_json.write_text(
                            json.dumps(summarize(rows), indent=2, sort_keys=True), encoding="utf-8"
                        )

            else:
                print(
                    f"[{case_name}] n={n}: direct method skipped; "
                    f"estimate {estimate / 1024**3:.2f} GiB exceeds limits or max n reached",
                    flush=True,
                )

    print("\nWrote:")
    for path in (raw_csv, summary_json, metadata_json, matrices_npz):
        print(f"  {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/benchmark_results")
    p.add_argument("--n-values", default="2:30")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--cold-repeats", type=int, default=1)
    p.add_argument("--direct-repeats", type=int, default=1)
    p.add_argument("--direct-warmups", type=int, default=0)
    p.add_argument("--direct-memory-gib", type=float, default=20.0)
    p.add_argument("--direct-max-n", type=int, default=9)
    p.add_argument("--threads", type=int, default=1)

    # Internal fresh-process worker options.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--case", choices=("two_term", "three_term"), help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.threads < 1:
        parser().error("--threads must be at least 1")
    if arguments.worker:
        if arguments.case is None or arguments.n is None:
            parser().error("worker mode requires --case and --n")
        raise SystemExit(cold_worker(arguments))
    raise SystemExit(main(arguments))
