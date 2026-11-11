#!/usr/bin/env python3
r"""Compile a PETSc tutorial, run its native simulation, and save CSR systems.

Example (from kband-spai/):
    python data_generators/petsc/generate.py --config data_generators/petsc/configs/burgers.json
    python data_generators/petsc/generate.py --config data_generators/petsc/configs/burgers.json \
        --output-dir data/petsc/burgers_fine --set da_grid_x=4095 --steps every:2

PETSc options are passed directly to the C executable. Only build copies are
instrumented; upstream sources and historical experiment scripts stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path: Path) -> dict:
    """Read a single tutorial configuration; PETSc options remain raw tokens."""
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    for field in ("name", "source", "source_version", "options"):
        if field not in config:
            raise ValueError(f"Missing configuration field: {field}")
    if not isinstance(config["name"], str) or not re.fullmatch(r"[\w.-]+", config["name"]) or config["name"] in {".", ".."}:
        raise ValueError("Configuration name must be a simple directory name")
    sources = [config["source"], *config.get("extra_sources", [])]
    for source in sources:
        if not isinstance(source, str) or Path(source).is_absolute() or ".." in Path(source).parts:
            raise ValueError("Sources must be relative paths inside upstream/")
        if not (HERE / "upstream" / source).is_file():
            raise FileNotFoundError(f"Tutorial source not found: {source}")
    if not isinstance(config["options"], list) or not all(isinstance(item, str) for item in config["options"]):
        raise ValueError("options must be a list of command-line strings")
    return config


def petsc_options(options: list[str], overrides: list[str], extra: list[str]) -> list[str]:
    """Append overrides in order; PETSc resolves repeated options last."""
    result = list(options)
    for override in overrides:
        key, separator, value = override.partition("=")
        key = key.lstrip("-")
        if not re.fullmatch(r"[A-Za-z_][\w-]*", key):
            raise ValueError(f"Invalid --set option: {override!r}; use NAME=VALUE or NAME")
        result.append("-" + key)
        if separator:
            if not value:
                raise ValueError("Use --set NAME for a PETSc flag without a value")
            result.append(value)
    result.extend(extra)
    if any(token.split("=", 1)[0] in {"-export_dir", "-export_steps", "-export_every"} for token in result):
        raise ValueError("Configure export options with --steps and --output-dir")
    return result


def sampling_options(selection: str | list[int]) -> list[str]:
    """Select the first linear solve on all, periodic, or listed TS steps."""
    if selection == "all":
        return ["-export_every", "1"]
    if isinstance(selection, str) and selection.startswith("every:"):
        interval = int(selection.split(":", 1)[1])
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")
        return ["-export_every", str(interval)]
    if isinstance(selection, str):
        selection = [int(value) for value in selection.split(",")]
    if not isinstance(selection, list) or not selection or any(type(step) is not int or step < 0 for step in selection):
        raise ValueError("Steps must be 'all', 'every:N', or a list of nonnegative integers")
    if len(selection) > 4096:
        raise ValueError("At most 4096 explicit sample steps; use every:N for longer sequences")
    return ["-export_steps", ",".join(map(str, sorted(set(selection))))]


def petsc_installation(prefix: Path):
    """Use this PETSc installation's binary reader and actual integer width."""
    conf = (prefix / "include/petscconf.h").read_text()
    defined = lambda name: bool(re.search(rf"^#define\s+{name}\s+1\s*$", conf, re.MULTILINE))
    if defined("PETSC_USE_COMPLEX") or not defined("PETSC_USE_REAL_DOUBLE"):
        raise ValueError("This generator requires a real, double-precision PETSc build")
    version_header = (prefix / "include/petscversion.h").read_text()
    version = ".".join(re.search(rf"#define\s+PETSC_VERSION_{field}\s+(\d+)", version_header).group(1)
                       for field in ("MAJOR", "MINOR", "SUBMINOR"))
    reader_path = prefix / "lib/petsc/bin/PetscBinaryIO.py"
    spec = importlib.util.spec_from_file_location("generator_petsc_binary", reader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index_width = "64bit" if defined("PETSC_USE_64BIT_INDICES") else "32bit"
    reader = module.PetscBinaryIO(precision="double", indices=index_width, complexscalars=False)
    return reader, {"prefix": str(prefix), "version": version, "precision": "double", "indices": index_width}


def instrument_source(text: str, source: str) -> tuple[str, list[str]]:
    """Apply documented compatibility fixes, then attach around TSSolve."""
    changes = []
    if source == "ex14.c":
        for local, global_vector in (("X2", "X2g"), ("F2", "F2g")):
            original = (
                f"  PetscCall(DMLocalToGlobalBegin(da2, {local}, INSERT_VALUES, {global_vector}));\n"
                f"  PetscCall(DMLocalToGlobalEnd(da2, {local}, INSERT_VALUES, {global_vector}));"
            )
            replacement = f"""  /* Serial-periodic DMDA INSERT compatibility: copy owned entries once. */
  {{
    const PrmNode **local_prm;
    PrmNode       **global_prm;
    PetscInt        gx, gy, gxm, gym, ii, jj;
    PetscCall(DMDAGetCorners(da2, &gx, &gy, NULL, &gxm, &gym, NULL));
    PetscCall(DMDAVecGetArrayRead(da2, {local}, &local_prm));
    PetscCall(DMDAVecGetArray(da2, {global_vector}, &global_prm));
    for (jj = gy; jj < gy + gym; jj++) {{
      for (ii = gx; ii < gx + gxm; ii++) global_prm[jj][ii] = local_prm[jj][ii];
    }}
    PetscCall(DMDAVecRestoreArray(da2, {global_vector}, &global_prm));
    PetscCall(DMDAVecRestoreArrayRead(da2, {local}, &local_prm));
  }}"""
            if text.count(original) != 1:
                raise ValueError(f"Unexpected IceFlow compatibility site: {local}")
            text = text.replace(original, replacement, 1)
        changes.append("IceFlow: replace two serial periodic DMDA INSERT transfers by owned-entry copies; preserve 3D ADD assembly")
    expression = r"(?m)^([ \t]*)PetscCall\(TSSolve\((\w+),[^\n]*\)\);"
    def attach(match):
        indent, ts = match.group(1), match.group(2)
        return (f"{indent}PetscCall(PetscExportAttach({ts}));\n" + match[0]
                + f"\n{indent}PetscCall(PetscExportFinish({ts}));")
    text, count = re.subn(expression, attach, text)
    if count != 1:
        raise ValueError(f"Expected one single-line TSSolve in {source}, found {count}")
    return f'#include "{HERE / "export.h"}"\n' + text, changes


def compile_tutorial(config: dict, prefix: Path, build_root: Path, env: dict, runtime: dict) -> tuple[Path, dict]:
    """Cache a native executable by source, export hook, and PETSc installation."""
    source = config["source"]
    upstream = HERE / "upstream"
    text, changes = instrument_source((upstream / source).read_text(), source)
    sources = [source, *config.get("extra_sources", [])]
    provenance_path = upstream / "sources.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    info = {"sources": {name: {"sha256": sha256(upstream / name), "provenance": provenance.get(name)} for name in sources},
            "source_version": config["source_version"], "runtime": runtime,
            "instrumented_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "export_header_sha256": sha256(HERE / "export.h"), "compatibility_changes": changes}
    compiler = prefix / "bin/mpicc"
    if not compiler.is_file():
        raise FileNotFoundError(f"MPI C compiler not found: {compiler}; select the PETSc environment with --petsc-prefix")
    info["compiler"] = str(compiler)
    info["compiler_environment"] = {key: env.get(key) for key in ("MPICH_CC", "CC", "CFLAGS")}
    key = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:20]
    build = build_root / Path(source).stem / key
    executable = build / "simulation"
    record_path = build / "build.json"
    if executable.is_file() and record_path.is_file():
        record = json.loads(record_path.read_text())
        return executable, {**record, "cache_hit": True}
    build.mkdir(parents=True, exist_ok=True)
    target = build / Path(source).name
    target.write_text(text)
    command = [str(compiler), "-O2", f"-I{prefix / 'include'}", f"-I{(upstream / source).parent}",
               str(target), *(str(upstream / item) for item in sources[1:]),
               f"-L{prefix / 'lib'}", f"-Wl,-rpath,{prefix / 'lib'}", "-lpetsc", "-lm", "-o", str(executable)]
    info.update(command=command, directory=str(build))
    print(f"Compiling {source}", flush=True)
    with (build / "build.log").open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"Compilation failed; see {build / 'build.log'}")
    write_json(record_path, info)
    return executable, {**info, "cache_hit": False}


def convert_systems(folder: Path, reader, keep_binary: bool) -> list[dict]:
    """Preserve exported A, b and effective native x0 in compressed CSR bundles."""
    metadata = folder / "systems.jsonl"
    exported = [json.loads(line) for line in metadata.read_text().splitlines()] if metadata.exists() else []
    if len({item["step"] for item in exported}) != len(exported):
        raise ValueError("Expected at most one exported KSP solve per TS step")
    for native in exported:
        raw = folder / f"system_{native['ordinal']:06d}.bin"
        objects = reader.readBinaryFile(str(raw))
        if len(objects) != 3:
            raise ValueError(f"Expected A, b and x0 in {raw}")
        shape, (indptr, indices, values) = objects[0]
        A = csr_matrix((np.array(values, dtype=np.float64), np.array(indices, dtype=np.int64),
                        np.array(indptr, dtype=np.int64)), shape=shape)
        A.check_format(full_check=True)
        A.sum_duplicates()
        A.sort_indices()
        b, x0 = (np.array(vector, dtype=np.float64) for vector in objects[1:])
        if A.shape != (b.size, b.size) or x0.shape != b.shape or b.ndim != 1:
            raise ValueError(f"Incompatible operator/RHS/guess shapes in {raw}")
        if not all(np.isfinite(array).all() for array in (A.data, b, x0)):
            raise ValueError(f"Nonfinite matrix, RHS or initial guess in {raw}")
        destination = raw.with_suffix(".npz")
        np.savez_compressed(destination, data=A.data, indices=A.indices, indptr=A.indptr,
                            shape=A.shape, b=b, native_x0=x0)
        native.update(path=destination.name, shape=list(A.shape), n=A.shape[0], nnz=A.nnz,
                      rhs_norm=float(np.linalg.norm(b)), sha256=sha256(destination))
        if not keep_binary:
            raw.unlink()
    return exported


def completion_status(returncode: int | None, completion: dict, exported: list[dict], sampling: list[str]) -> tuple[str, list[str]]:
    """A clean sequence needs successful native convergence and requested samples."""
    problems = []
    if returncode != 0:
        problems.append(f"Native process failed or timed out (returncode={returncode})")
    if completion.get("ts_converged_reason", 0) <= 0:
        problems.append("Native time integrator did not report successful convergence")
    if not exported:
        problems.append("No linear systems exported; use an implicit formulation with assembled operators")
    if sampling[0] == "-export_steps":
        missing = set(map(int, sampling[1].split(","))) - {item["step"] for item in exported}
        if missing:
            problems.append(f"Requested TS steps not exported: {sorted(missing)}; adjust --steps or simulation duration")
    if problems:
        return "incomplete", problems
    if completion.get("step_rejections", 0) or completion.get("snes_failures", 0):
        return "rejected_attempts", ["Trajectory had rejected steps or SNES failures; snapshots can precede rejected attempts"]
    return "complete", []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration for one tutorial")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; default: data/petsc/<config name>")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "results/petsc_build", help="Executable cache and compilation logs")
    parser.add_argument("--petsc-prefix", type=Path, default=Path(sys.prefix), help="Installed PETSc/MPI prefix; default: active Python environment")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Append a PETSc option override; repeat as needed; NAME alone sets a flag")
    parser.add_argument("--steps", help="TS steps to capture: all, every:N, or 0,2,5; default: config export_steps")
    parser.add_argument("--max-seconds", type=float, default=300, help="Native run timeout (seconds)")
    parser.add_argument("--keep-binary", action="store_true", help="Retain temporary PETSc binary bundles after NPZ conversion")
    parser.add_argument("--petsc", nargs=argparse.REMAINDER, default=[], help="Pass remaining raw arguments to PETSc; place this last")
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
        options = petsc_options(config["options"], args.set, args.petsc)
        sampling = sampling_options(args.steps if args.steps is not None else config.get("export_steps", "all"))
        if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
            raise ValueError("--max-seconds must be finite and positive")
        reader, runtime = petsc_installation(args.petsc_prefix.resolve())
        output = (args.output_dir or ROOT / "data/petsc" / config["name"]).resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Generation setup failed: {error}\n")

    # Serial local MPI execution; respect existing compiler/transport/thread settings.
    env = dict(os.environ)
    for name, value in {"MPICH_CC": "gcc", "UCX_TLS": "self", "UCX_NET_DEVICES": "none", "UCX_LOG_LEVEL": "error",
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
        env.setdefault(name, value)
    record = {"configuration": config, "config_path": str(args.config.resolve()), "runtime": runtime,
              "config_sha256": sha256(args.config.resolve()), "generator_sha256": sha256(Path(__file__).resolve()),
              "invocation": [sys.executable, *sys.argv],
              "output_directory": str(output), "sampling": sampling, "effective_options": options,
              "export": "First KSP solve per selected TS step; actual operator A, RHS b, effective native initial guess",
              "environment": {name: env.get(name) for name in ("MPICH_CC", "UCX_TLS", "UCX_NET_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "status": "error", "systems": []}
    started = time.perf_counter()
    try:
        executable, compilation = compile_tutorial(config, args.petsc_prefix.resolve(), args.build_dir.resolve(), env, runtime)
        record["compilation"] = compilation
        command = [str(executable), *options, "-export_dir", str(output), *sampling]
        record["command"] = command
        print(f"Running: {shlex.join(command)}", flush=True)
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            try:
                process = subprocess.run(command, cwd=output, env=env, stdout=stdout, stderr=stderr, timeout=args.max_seconds)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = None
                record["timeout_seconds"] = args.max_seconds
        record["returncode"] = returncode
        native_path = output / "run.json"
        completion = json.loads(native_path.read_text()) if native_path.exists() else {}
        record["native_completion"] = completion
        record["systems"] = convert_systems(output, reader, args.keep_binary)
        record["status"], record["problems"] = completion_status(returncode, completion, record["systems"], sampling)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        print(record["error"], file=sys.stderr)
    record["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "manifest.json", record)
    print(f"{config['name']}: {record['status']}; systems={len(record['systems'])}; output={output}", flush=True)
    for problem in record.get("problems", []):
        print(problem, file=sys.stderr)
    if record["status"] != "complete":
        print(f"Inspect {output / 'manifest.json'} and stdout.log/stderr.log for details", file=sys.stderr)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
