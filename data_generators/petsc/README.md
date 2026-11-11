# PETSc Linear-System Generation Manual

Run native PETSc C simulations and save assembled linear systems for Python
experiments. PETSc and the compiler come from your environment.

```text
petsc/
├── generate.py
├── export.h
├── configs/        # One JSON configuration per simulation
├── upstream/       # Unmodified PETSc C sources, provenance, and license
└── README.md
```

Workflow: **JSON → instrumented C build → native PETSc simulation → binary
systems → CSR `.npz` files**. Python compiles with `mpicc`; `export.h` captures
systems before selected linear solves. Bundled sources allow offline runs.

## Requirements and Quick Start

Requires Python, NumPy, SciPy, PETSc headers/libraries, `mpicc`, and a C
compiler. Use the existing `sc` environment. Generation uses **one MPI rank
on CPU**, with real double PETSc scalars and automatic integer-width
detection. No SPAI installation, `PYTHONPATH`, CUDA, or petsc4py is needed.

Validated environment: **PETSc 3.25.4 / MPICH**, with bundled tutorial sources
from **PETSc v3.24.0**. Other installations have not been validated.

From `kband-spai/`:

```bash
conda activate sc
python data_generators/petsc/generate.py --config data_generators/petsc/configs/burgers.json
python data_generators/petsc/generate.py --config data_generators/petsc/configs/radiation_blast.json
python data_generators/petsc/generate.py --config data_generators/petsc/configs/iceflow.json
python data_generators/petsc/generate.py --config data_generators/petsc/configs/allen_cahn.json
```

Defaults advance **six timesteps**, exporting steps **0, 2, 5** to
`data/petsc/<name>/`. Build caches/logs use `results/petsc_build/`; both
locations are Gitignored. Each run requires a fresh output directory.

## Simulations and Dimensions

| Configuration | C source | Resolution options | Default matrix dimension |
| --- | --- | --- | --- |
| [Burgers](configs/burgers.json) | [ex9.c](upstream/ex9.c) | `-da_grid_x` | 1025 |
| [Radiation blast](configs/radiation_blast.json) | [ex10.c](upstream/ex10.c) | `-da_grid_x` | 1024 = 2 × 512 |
| [Native IceFlow](configs/iceflow.json) | [ex14.c](upstream/ex14.c) | `-M`, `-N`, `-P` | 2160 = 12 × 12 × (2 × 6 + 3) |
| [Varying-coefficient Allen–Cahn](configs/allen_cahn.json) | [ex29.c](upstream/ex29.c) | `-da_grid_x`, `-da_grid_y` | 2401 = 49 × 49 |

These configurations produce nonsymmetric systems. The manifest records
actual dimensions; changing the formulation can change the formulas above.

## Changing Parameters

Copy a tested configuration, then edit its fields:

```bash
cp data_generators/petsc/configs/radiation_blast.json \
   data_generators/petsc/configs/radiation_custom.json
```

| JSON field | Meaning |
| --- | --- |
| `name` | Required; simple default output directory name. |
| `source` | Required; tutorial path relative to bundled `upstream/`. |
| `source_version` | Required; release label, currently `v3.24.0`. |
| `options` | Required; ordered argument strings, e.g. `["-ts_dt", "0.001", "-snes_fd_color"]`. |
| `export_steps` | Optional; integer list, `"all"`, or `"every:N"`; defaults to `"all"`. |
| `extra_sources` | Optional; additional C filenames relative to `upstream/`. |

Options pass directly to PETSc, preserving upstream runtime flexibility.
Hardcoded choices still require C changes.

For overrides, repeat `--set NAME=VALUE`. Use `--set NAME` for a flag;
negative values work as `--set NAME=-1`. Precedence is **JSON options →
repeated `--set` options → raw `--petsc` arguments**; the last occurrence wins:

```bash
python data_generators/petsc/generate.py \
  --config data_generators/petsc/configs/burgers.json \
  --output-dir data/petsc/burgers_4095 \
  --set da_grid_x=4095 --set ts_dt=0.005 --set ts_max_steps=100 \
  --steps every:5
```

For IceFlow use, for example, `--set M=16 --set N=16 --set P=8`.
Raw argument lists are also supported after `--petsc`, which must come last:

```bash
python data_generators/petsc/generate.py \
  --config data_generators/petsc/configs/radiation_blast.json \
  --output-dir data/petsc/radiation_1024 \
  --steps all --petsc -da_grid_x 1024 -ts_monitor -snes_converged_reason
```

`--petsc -help` writes native help into `stdout.log` **and runs the simulation**;
`--petsc -options_left` reports unused options. Generator `--help` only
displays help.

### Other Generator Options and Paths

| Option | Purpose / default |
| --- | --- |
| `--output-dir PATH` | Fresh destination; default `kband-spai/data/petsc/<name>/`. |
| `--build-dir PATH` | Compilation cache/logs; default `kband-spai/results/petsc_build/`. |
| `--petsc-prefix PATH` | Installed PETSc/MPI prefix; defaults to the active Python environment. |
| `--max-seconds N` | Native simulation timeout; default 300 seconds. Compilation has a separate 120-second limit. |
| `--keep-binary` | Keep PETSc binary bundles alongside converted `.npz` files. |

Explicit CLI paths resolve from your current directory; source paths resolve
inside bundled `upstream/`. Native PETSc paths resolve inside the output
directory: use absolute external-input paths. Default data/build locations
remain relative to `kband-spai/`. Runtime-only changes reuse cached builds;
the manifest records cache usage.

## Sampling and Output

Use `--steps all`, `--steps every:10` (steps 0, 10, 20, …), or `--steps 0,2,5`.
The equivalent JSON `export_steps` accepts these strings or an integer list
(up to 4096 entries). Six timesteps have capture labels 0–5. Only the **first
KSP call per selected TS step** is saved. With `all`/`every:N`, KSP-free steps
produce no snapshot. Filenames use KSP call ordinals; numbering gaps are normal.

Each output directory contains:

- `system_*.npz`: CSR arrays `data`, `indices`, `indptr`, `shape`, plus RHS `b`
  and effective native initial guess `native_x0`.
- `manifest.json`: configuration, effective command, source/header hashes,
  compilation details, PETSc version, completion status and system metadata.
- `systems.jsonl` and `run.json`: metadata emitted by the C export hook.
- `stdout.log` and `stderr.log`: native simulation output.

Load a saved system:

```python
import numpy as np
from scipy.sparse import csr_matrix

with np.load("data/petsc/burgers/system_000000.npz", allow_pickle=False) as saved:
    A = csr_matrix((saved["data"], saved["indices"], saved["indptr"]),
                   shape=tuple(saved["shape"]))
    b = saved["b"]
    x0 = saved["native_x0"]
```

Python removes converted binaries unless `--keep-binary` is set. Native
visualization files, if produced, stay in the output directory.

## Numerical Meaning and Completion

PETSc's TS/SNES/KSP handle timestepping/nonlinear/linear solves. Defaults
advance with LU; SPAI comparisons replay saved systems separately. Exports
contain the actual KSP operator through a CSR copy. Nonlinear `A, b` generally
describe Newton correction equations; `native_x0` is the linear initial
guess. Metadata times are TS-reported capture times.

Clean completion requires process success, positive TS convergence reason,
at least one exported system and every explicitly requested sample.

Snapshots can precede rejected attempts and are not replaced by retries.
Rejections/SNES failures mark a run `rejected_attempts`. Supplied configurations
disable timestep rejection.

## Troubleshooting and Exit Status

Exit codes: **0** = complete; **1** = build, run or conversion failure,
incomplete output or rejected trajectory; **2** = argument/configuration/setup
failure. Failed-run artifacts remain available for inspection.

| Symptom | Action |
| --- | --- |
| Output directory exists | Choose a fresh `--output-dir`; existing directories are never overwritten. |
| Missing PETSc files or `mpicc` | Activate the PETSc environment or set `--petsc-prefix`; inspect `build.log` for compilation errors. |
| No systems exported | Use an implicit formulation invoking KSP. Explicit-only steps may produce no linear solves. |
| Matrix-free operator rejected | Use an assembled Jacobian with accessible entries. |
| Requested steps missing | Extend the run or reduce `--steps`; early termination/KSP-free steps can omit samples. |
| Timeout / nonlinear failure | Inspect logs and manifest problems; adjust timeout, timestep or solver settings. |
| Burgers coloring error | With the supplied coloring settings, choose `da_grid_x` divisible by five. |

For longer sequences, check both `-ts_max_steps` and `-ts_max_time`.

## Source Details and Adding Tutorials

[`upstream/sources.json`](upstream/sources.json) records official URLs, versions
and SHA256 hashes; [`upstream/LICENSE`](upstream/LICENSE) preserves PETSc's
license. Preserve originals; transform build copies. Manifests record actual
and provenance hashes without enforcing a match. IceFlow's build copy
receives the documented serial-periodic DMDA owned-entry compatibility fix.

Burgers retains `-snes_fd_color` to form the full residual Jacobian. Its
periodic coloring requires grid counts divisible by five, e.g. 125/1025/4095.
Radiation `-rd_initial 2` selects blast; `1` selects Marshak wave.

To add a tutorial, place pinned sources in `upstream/`, record provenance,
and add JSON. Instrumentation expects one single-line
`PetscCall(TSSolve(...));`; other structures, dependencies or compatibility
fixes need explicit extensions to `generate.py`. Keep historical screening
drivers for reproduction.

## Validation

Run the focused tests from `kband-spai/`:

```bash
python -m pytest -q tests/test_petsc_generator.py
```

The 49 focused tests cover sources, instrumentation, options, sampling,
completion and conversion without native compilation. Initial CPU validation:
eight native runs, 22 systems, worst independent sparse-LU residual `5.1e-12`.
The [local validation record](../../results/petsc_generator_validation/validation.json)
is ignored and absent from fresh checkouts. Validate changes with short native
runs, actual dimensions and explicit residuals before long sequences;
do not calculate eigenvalues.
