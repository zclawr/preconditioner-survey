import sys
import petsc4py
import operators.poisson
import argparse

petsc4py.init(sys.argv)

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

def linear_solve(A):
    A.viewFromOptions('-view_mat')
    # PETSc represents all linear solvers as preconditioned Krylov subspace methods
    # of type `PETSc.KSP`. Here we create a KSP object for a conjugate gradient
    # solver preconditioned with an algebraic multigrid method.
    ksp = PETSc.KSP()
    ksp.create(comm=A.getComm())
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.GAMG)

    # We set the matrix in our linear solver and allow the user to program the
    # solver with options.
    ksp.setOperators(A)
    ksp.setFromOptions()

    # Since the matrix knows its size and parallel distribution, we can retrieve
    # appropriately-scaled vectors using `Mat.createVecs`. PETSc vectors are
    # objects of type `PETSc.Vec`. Here we set the right-hand side of our system to
    # a vector of ones, and then solve.
    x, b = A.createVecs()
    b.set(1.0)
    ksp.solve(b, x)

    x.viewFromOptions('-view_sol')

operator_map = {
    "poisson": operators.poisson.construct_poisson_operator    
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure preconditioner survey settings")
    parser.add_argument("operator", help=r"Available operators: {poisson}")

    args, petsc_args = parser.parse_known_args()
    op_name = args.operator
    op = operator_map[op_name]

    A,n,h = init()
    A = op(A,n,h)
    linear_solve(A)
