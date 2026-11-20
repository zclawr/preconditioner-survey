"""Baseline inverse preconditioners and SciPy/PyTorch CSR conversion.

Explicit constructors return SciPy CSR matrices ``P`` applied as ``P @ r``.
ILU and optional IC(0) return inverse-action ``LinearOperator`` instances.
These constructors do not choose the physical Krylov solver: ordinary CG
requires an SPD system and an SPD preconditioner; ILU generally needs GMRES.
"""

from __future__ import annotations

from collections import defaultdict
from operator import index
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator, spilu
import torch

__all__ = [
    "scipy_to_torch_csr", "torch_csr_to_scipy", "identity", "jacobi",
    "block_jacobi", "richardson_jacobi", "generate_sparsity_pattern",
    "compute_spai", "ilu", "incomplete_cholesky",
]


def _square_csr(A) -> sp.csr_matrix:
    if isinstance(A, torch.Tensor):
        A = torch_csr_to_scipy(A)
    result = sp.csr_matrix(A)
    n, m = result.shape
    if n == 0 or n != m:
        raise ValueError("A must be nonempty and square.")
    if not np.issubdtype(result.dtype, np.inexact):
        result = result.astype(np.float64)
    if not np.isfinite(result.data).all():
        raise ValueError("A must have finite entries.")
    return result


def scipy_to_torch_csr(
    A: sp.spmatrix, device: torch.device | str = "cpu", dtype=torch.float64
) -> torch.Tensor:
    """Copy a SciPy sparse matrix to canonical torch CSR on an explicit device.

    The CPU default makes imports and small reference runs independent of CUDA.
    Complex input requires a complex ``dtype`` to prevent loss of its imaginary
    component. Unlike inverse-preconditioner constructors, conversion may be
    used with rectangular matrices.
    """
    A = sp.csr_matrix(A, copy=True)
    A.sum_duplicates()
    A.sort_indices()
    if np.iscomplexobj(A.data) and not dtype.is_complex:
        raise TypeError("Complex input requires a complex torch dtype.")
    if not (dtype.is_floating_point or dtype.is_complex):
        raise TypeError("dtype must be floating point or complex.")
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64, copy=True)),
        torch.from_numpy(A.indices.astype(np.int64, copy=True)),
        torch.from_numpy(A.data.copy()).to(dtype=dtype),
        size=A.shape, device=device,
    )


def torch_csr_to_scipy(A_torch: torch.Tensor, dtype=None) -> sp.csr_matrix:
    """Copy torch CSR to SciPy CSR; a CUDA input is explicitly moved to CPU."""
    if not isinstance(A_torch, torch.Tensor) or not A_torch.is_sparse_csr:
        raise TypeError("A_torch must be a torch sparse CSR tensor.")
    A_cpu = A_torch.detach().cpu()
    result = sp.csr_matrix(
        (A_cpu.values().numpy().copy(), A_cpu.col_indices().numpy().copy(),
         A_cpu.crow_indices().numpy().copy()),
        shape=tuple(A_cpu.shape),
    )
    return result.astype(dtype) if dtype is not None else result


def identity(A_or_n, *, dtype=None) -> sp.csr_matrix:
    """Identity inverse action: the unpreconditioned baseline."""
    if hasattr(A_or_n, "shape"):
        A = _square_csr(A_or_n)
        n = A.shape[0]
        dtype = A.dtype if dtype is None else dtype
    else:
        n = index(A_or_n)
        if n <= 0:
            raise ValueError("n must be positive.")
    return sp.eye(n, dtype=np.float64 if dtype is None else dtype, format="csr")


def jacobi(A) -> sp.csr_matrix:
    """Return ``diag(A)**-1``; missing or zero diagonal entries are errors."""
    diagonal = _square_csr(A).diagonal()
    if np.any(diagonal == 0):
        raise ValueError("Jacobi requires nonzero diagonal entries.")
    return sp.diags(1.0 / diagonal, format="csr")


def block_jacobi(A, block_size: int = 32) -> sp.csr_matrix:
    """Invert consecutive diagonal blocks, including a final smaller block.

    Only individual blocks are densified. A singular block raises
    ``numpy.linalg.LinAlgError`` identifying its row range.
    """
    block_size = index(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    A = _square_csr(A)
    blocks = []
    for start in range(0, A.shape[0], block_size):
        stop = min(start + block_size, A.shape[0])
        try:
            blocks.append(np.linalg.inv(A[start:stop, start:stop].toarray()))
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"Diagonal block [{start}:{stop}] is singular."
            ) from exc
    result = sp.block_diag(blocks, format="csr")
    result.eliminate_zeros()
    return result


def richardson_jacobi(A, *, omega: float = 1.0) -> sp.csr_matrix:
    r"""Two Jacobi/Richardson steps: ``2 omega J - omega**2 J A J``.

    This preserves the original ``2J-JAJ`` baseline for ``omega=1``. Positivity
    for SPD ``A`` additionally requires ``omega*lambda_max(J**.5 A J**.5)<2``.
    """
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive.")
    A = _square_csr(A)
    J = jacobi(A)
    return (2.0 * omega * J - omega**2 * (J @ A @ J)).tocsr()


def generate_sparsity_pattern(
    n: int, offsets: Iterable[int], *, symmetric_pattern: bool = True
) -> list[np.ndarray]:
    """Allowed row indices in each SPAI column; the diagonal is always present.

    Positive signed offset ``d`` means ``M[i, i+d]``. By default offsets are
    mirrored using their absolute values, matching legacy ``compute_spai``.
    Mirroring support does *not* impose equal coefficients on opposite bands.
    """
    n = index(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    distances = {index(d) for d in offsets}
    if any(abs(d) >= n for d in distances):
        raise ValueError("Offsets must satisfy abs(d) < n.")
    if symmetric_pattern:
        distances |= {-d for d in distances}
    distances.add(0)
    return [np.array(sorted(k - d for d in distances if 0 <= k - d < n),
                     dtype=np.int64) for k in range(n)]


def compute_spai(
    A, offsets: Iterable[int], dtype=torch.float64, driver: str | None = None,
    *, device: torch.device | str = "cpu", objective: str = "AM",
    symmetric_pattern: bool = True, max_batch_bytes: int = 256 * 1024**2,
) -> sp.csr_matrix:
    r"""Direct fixed-pattern least-squares SPAI, with independent coefficients.

    Default objective is ``min ||A M-I||_F`` and mirrored band *support*, as in
    the original implementation. ``objective='MA'`` fits ``||M A-I||_F``.
    Use ``symmetric_pattern=False`` for independent signed support; positive
    offsets denote upper bands of the returned ``M`` for either objective.

    Sparse slicing stays on CPU. Local dense least-squares systems are grouped
    by shape and solved on ``device`` under an approximate batching budget.
    No process pool or implicit device selection is used. The result is SciPy
    CSR on CPU; use ``scipy_to_torch_csr`` for a GPU inverse action. CPU uses
    rank-revealing ``gelsy`` by default; CUDA supports only ``gels``, requiring
    full-column-rank local problems.
    """
    A = _square_csr(A)
    offsets = tuple(index(d) for d in offsets)
    if objective not in ("AM", "MA"):
        raise ValueError('objective must be "AM" or "MA".')
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive.")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Direct SPAI currently supports float32 and float64.")
    if np.iscomplexobj(A.data):
        raise TypeError("Direct SPAI currently supports real matrices only.")
    if objective == "MA":
        return compute_spai(
            A.T, tuple(-d for d in offsets), dtype=dtype, driver=driver,
            device=device, objective="AM", symmetric_pattern=symmetric_pattern,
            max_batch_bytes=max_batch_bytes,
        ).T.tocsr()

    device = torch.device(device)
    if driver is None:
        driver = "gels" if device.type == "cuda" else "gelsy"
    if device.type == "cuda" and driver != "gels":
        raise ValueError('CUDA least squares supports driver="gels" only.')
    n = A.shape[0]
    patterns = generate_sparsity_pattern(n, offsets, symmetric_pattern=symmetric_pattern)
    A_csc = A.tocsc()
    groups = defaultdict(list)
    for k, pattern in enumerate(patterns):
        submatrix = A_csc[:, pattern]
        active_rows = np.unique(submatrix.indices)
        groups[(len(active_rows), len(pattern))].append((k, active_rows))

    pointers = np.zeros(n + 1, dtype=np.int64)
    pointers[1:] = np.cumsum([len(pattern) for pattern in patterns])
    out_indices = np.concatenate(patterns)
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    out_values = np.zeros(pointers[-1], dtype=numpy_dtype)
    itemsize = np.dtype(numpy_dtype).itemsize
    for (rows, cols), group in groups.items():
        if rows == 0:
            continue  # The zero local operator has minimum-norm solution zero.
        per_problem = itemsize * max(1, 3 * rows * cols + rows + 2 * cols)
        batch_size = max(1, max_batch_bytes // per_problem)
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            local_A = np.empty((len(batch), rows, cols), dtype=numpy_dtype)
            local_b = np.zeros((len(batch), rows, 1), dtype=numpy_dtype)
            for j, (k, active_rows) in enumerate(batch):
                local_A[j] = A_csc[:, patterns[k]][active_rows, :].toarray()
                position = np.searchsorted(active_rows, k)
                if position < rows and active_rows[position] == k:
                    local_b[j, position, 0] = 1.0
            solutions = torch.linalg.lstsq(
                torch.as_tensor(local_A, device=device),
                torch.as_tensor(local_b, device=device), driver=driver,
            ).solution.detach().cpu().numpy()
            for j, (k, _) in enumerate(batch):
                out_values[pointers[k]:pointers[k + 1]] = solutions[j, :, 0]
    return sp.csc_matrix((out_values, out_indices, pointers), shape=(n, n)).tocsr()


def ilu(A, **spilu_options) -> LinearOperator:
    """Return a SciPy SuperLU incomplete-LU inverse action for GMRES.

    Forward ``drop_tol``, ``fill_factor`` and related options to ``spilu``.
    The returned operator applies triangular solves; no dense inverse is built.
    """
    A = _square_csr(A)
    factor = spilu(A.tocsc(), **spilu_options)
    return LinearOperator(
        A.shape, matvec=factor.solve, matmat=factor.solve,
        rmatvec=lambda r: factor.solve(r, trans="H"), dtype=A.dtype,
    )


def incomplete_cholesky(A, **options) -> LinearOperator:
    """Return optional ``ilupp.IChol0Preconditioner`` as an inverse action.

    Requires the separately installed ``ilupp`` package and real SPD input.
    Importing this module does not import or require ilupp. No fallback to ILU
    is made, since a general ILU action is not necessarily SPD.
    """
    A = _square_csr(A)
    try:
        import ilupp
    except ImportError as exc:
        raise ImportError("Incomplete Cholesky requires the optional ilupp package.") from exc
    return aslinearoperator(ilupp.IChol0Preconditioner(A, **options))