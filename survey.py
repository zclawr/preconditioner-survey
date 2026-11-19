import argparse
import csv
import itertools
import os
import sys

import petsc4py

from solve import linear_solve, operator_map

DEFAULT_PCS = ["none", "jacobi", "sor", "ilu", "icc", "bjacobi", "asm", "gamg", "hypre"]

def run_survey(args):
    history = []
    op = operator_map[args.operator]
    for n in args.n:
        A, b = op(n)
        for pc in args.pcs:
            print(f'Solving {args.operator} with n={n} preconditioned by: {pc}')
            result = linear_solve(A,b, args.solver, pc, args.rtol, args.max_it, args.gmres_restart)
            history.append(result)
        # Clean up for memory management
        A.destroy()
        b.destroy()
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preconditioner survey settings")
    parser.add_argument("-op", "--operator", default="poisson")
    parser.add_argument("-n", "--n", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("-p", "--pcs", nargs="+", default=DEFAULT_PCS)
    parser.add_argument("-s", "--solver", default="gmres")
    parser.add_argument("-t", "--rtol", type=float, default=1e-8)
    parser.add_argument("-i", "--max-it", type=int, default=1000)
    parser.add_argument("-r", "--gmres-restart", type=int, default=20)
    parser.add_argument("-O", "--outdir", default="results")
    parser.add_argument("--plot", action="store_true")
    
    args, petsc_args = parser.parse_known_args(sys.argv[1:])
    petsc4py.init([sys.argv[0]] + petsc_args)

    from petsc4py import PETSc # must follow init, which requires arg parsing first

    results = run_survey(args)

   