import sys
import petsc4py
import argparse

import operators.poisson
from petsc4py import PETSc 

# Relevant commands
# -------------
# - Show the solution with ``-view_sol``.
# - Show the matrix with ``-view_mat``.
# - Change the resolution with ``-n``.
# - Use a direct solver by passing ``-ksp_type preonly -pc_type lu``.
# - Run in parallel on two processors using:
#
#   .. code-block:: console
#
#       mpiexec -n 2 python poisson2d.py

def init():
    OptDB = PETSc.Options()

    # The user can specify a different number of points in each direction by passing -n
    n = OptDB.getInt('n', 5)
    h = 1.0 / (n + 1)

    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)
    A.setSizes((n * n, n * n))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setFromOptions()

    # Insertion into some matrix types is vastly more efficient if we preallocate
    # space rather than allow this to happen dynamically. Here we hint the number
    # of nonzeros to be expected on each row.
    A.setPreallocationNNZ(5)

    return A,n,h

def linear_solve(A, b, solver="gmres", pc="gamg", rtol=1e-8, max_it=1000, gmres_restart=20):
    A.viewFromOptions('-view_mat')

    ksp = PETSc.KSP()
    ksp.create(comm=A.getComm())
    ksp.setType(solver)
    ksp.getPC().setType(pc)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    if "gmres" in solver:
        ksp.setGMRESRestart(gmres_restart)
    #TODO: Verify whether this is the correct norm type, this was a little confusing in the docs
    ksp.setNormType(PETSc.KSP.NormType.NORM_UNPRECONDITIONED)
    ksp.setInitialGuessNonzero(False)
    # Allow user to force options from cmd line
    ksp.setFromOptions()

    x = A.createVecs()[0]
    ksp.setUp()
    history = []

    def monitor(ksp, it, rnorm): # check docs for monitor specs if you'd like to edit this
        history.append((it, rnorm))

    ksp.setMonitor(monitor)

    ksp.solve(b, x)

    result = {
        "it": ksp.getIterationNumber(),
        "converged": ksp.getConvergedReason() > 0,
        "history": history
    }

    x.viewFromOptions('-view_sol')
    # Clean up for memory management
    x.destroy()
    ksp.destroy()
    return result

operator_map = {
    "poisson": operators.poisson.construct_poisson_operator    
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure preconditioner survey settings")
    parser.add_argument("operator", default='poisson', help="Available operators: {poisson}")
    parser.add_argument("-n", "--n", default=32, help="Operator width n (where A is n x n)")
    parser.add_argument("-s", "--solver", default='gmres', help="Available solvers: {GMRES, LGMRES, RGMRES, CG}") #TODO: verify available solver types in PETSc
    parser.add_argument("-p", "--pc", default='gamg', help="Available preconditioners: {ilu, icc, gamg, hypre}") #TODO: verify available preconditioners in PETSc
    parser.add_argument("-t", "--rtol", default=1e-8, help="Residual tolerance")
    parser.add_argument("-i", "--max_it", default=1000, help="Maximum solver iterations")
    parser.add_argument("-r", "--gmres_testart", default=20, help="GMRES restart iterations")

    args, petsc_args = parser.parse_known_args()
    petsc4py.init([sys.argv[0]] + petsc_args)

    op_name = args.operator
    op = operator_map[op_name]
    A, b = op(args.n)
    result = linear_solve(A, b, args.solver, args.pc, args.rtol, args.max_it, args.gmres_restart)

