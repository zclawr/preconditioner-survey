from util import index_to_grid
from petsc4py import PETSc

# Solve a constant coefficient Poisson problem on a regular grid.
#     - u_{xx} - u_{yy} = 1 \quad\textsf{in}\quad [0,1]^2\\
#     u = 0 \quad\textsf{on the boundary.}

def construct_poisson_operator(n):
    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)
    A.setSizes((n * n, n * n))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setFromOptions()

    # Insertion into some matrix types is vastly more efficient if we preallocate
    # space rather than allow this to happen dynamically. Here we hint the number
    # of nonzeros to be expected on each row.
    A.setPreallocationNNZ(5)

    h = 1.0 / (n + 1)
    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = index_to_grid(row,n)
        A[row, row] = 4.0 / h**2
        if i > 0:
            column = row - n
            A[row, column] = -1.0 / h**2
        if i < n - 1:
            column = row + n
            A[row, column] = -1.0 / h**2
        if j > 0:
            column = row - 1
            A[row, column] = -1.0 / h**2
        if j < n - 1:
            column = row + 1
            A[row, column] = -1.0 / h**2

    A.assemblyBegin()
    A.assemblyEnd()

    b = A.createVecs()[1]
    b.set(1.0)

    return A, b